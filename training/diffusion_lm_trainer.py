print("⚡ Thunder Trace: Script started, importing dependencies...", flush=True)
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm

print("⚡ Thunder Trace: Basic dependencies imported. Importing local modules...", flush=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import THUNDER_CONFIG
from core.model_loader import ThunderModelLoader
from core.diffusion_model import PrefixLMDiffusionAdapter
from training.noise_scheduler import ThunderNoiseScheduler
from training.loss_functions import DiffusionLMLoss
from training.data_pipeline import ThunderDataPipeline
print("⚡ Thunder Trace: All modules imported.", flush=True)

class DiffusionLMTrainer:
    """
    Custom training loop for continuous Diffusion-LM with LLaDA-8B (Bidirectional).
    We do NOT use SFTTrainer because we need tight control over:
    1. The continuous embedding bridge.
    2. The diffusion timestep sampling.
    3. The x0-parametrization MSE loss and L_round penalty.
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.noise_scheduler = ThunderNoiseScheduler()
        self.loss_fn = DiffusionLMLoss(t_round_penalty=config.get("training", {}).get("t_round_penalty", 0.05))
        
        self.device = self.model.device
        self.dtype = self.model.dtype

        # 1. Setup output directory and scan for existing checkpoints
        self.output_dir = self.config["training"].get("output_dir", "./thunder_diffusion_checkpoints")
        self.saved_checkpoints = []
        
        if os.path.exists(self.output_dir):
            import glob
            import re
            # Find all checkpoint directories
            existing = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
            # Sort by step number (e.g., checkpoint-500, checkpoint-1000)
            def extract_step(path):
                match = re.search(r"checkpoint-(\d+)", path)
                return int(match.group(1)) if match else 0
            
            existing.sort(key=extract_step)
            self.saved_checkpoints = existing
            if len(self.saved_checkpoints) > 0:
                print(f"⚡ Thunder: Detected {len(self.saved_checkpoints)} existing checkpoints in {self.output_dir}")

    def train(self, dataset):
        """
        Main training loop.
        """
        print(f"⚡ Thunder PrefixLM: Starting custom training loop on {self.device}...")
        
        # Prepare DataLoader
        batch_size = self.config["training"].get("batch_size", 4)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn
        )
        
        # Optimizer
        learning_rate = self.config["training"].get("learning_rate", 5e-5)
        # We want to optimize the base model (adapters) and our custom heads (x0_head, timestep_embedder)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        max_steps = self.config["training"].get("max_steps", None)
        epochs = self.config["training"].get("epochs", 3)
        if max_steps is not None:
            epochs = max(1, (max_steps + len(dataloader) - 1) // len(dataloader))
            num_training_steps = max_steps
        else:
            num_training_steps = epochs * len(dataloader)
        if max_steps is not None:
            num_training_steps = max_steps
        
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config["training"].get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )
        
        self.model.train()
        
        global_step = 0
        grad_accum_steps = self.config["training"].get("grad_accum", 1)
        
        print(f"⚡ Thunder: Training for {num_training_steps} total iterations (updates)...")
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # 1. Get exact clean embeddings (x0)
                embedding_matrix = self.model.get_input_embeddings().weight # [V, H]
                std_embedding_matrix = embedding_matrix
                
                clean_embeddings = self.model.get_input_embeddings()(input_ids) # [B, L, H]
                emb_scale = (clean_embeddings.size(-1) ** 0.5)
                standardized_x0 = clean_embeddings * emb_scale
                
                # 2. Sample random timesteps
                bsz = input_ids.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.diffusion_steps, (bsz,), 
                    device=self.device
                ).long()
                
                # 3. Add noise in the Standardized Space ($N(0, 1)$)
                noise = torch.randn_like(standardized_x0)
                noisy_latents = self.noise_scheduler.add_noise(standardized_x0, noise, timesteps)
                
                # Default device type for autocast
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'

                # Use Automatic Mixed Precision for Forward and Loss
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    # 4. Forward Pass (PrefixLM)
                    cfg_mask = attention_mask.clone()
                    cfg_drop_rate = self.config["diffusion"].get("cfg_drop_rate", 0.15)
                    if torch.rand(1).item() < cfg_drop_rate:
                        cfg_mask = torch.zeros_like(cfg_mask)

                    self_cond = None
                    if torch.rand(1).item() < 0.5:
                        with torch.no_grad():
                            self_cond = self.model.diffusion_forward(
                                x_t=noisy_latents, 
                                t=timesteps, 
                                attention_mask=cfg_mask
                            ).detach()
                    
                    x0_pred = self.model.diffusion_forward(
                        x_t=noisy_latents, 
                        t=timesteps, 
                        attention_mask=cfg_mask,
                        self_cond=self_cond
                    )
                    
                    # 5. Compute Diffusion Losses
                    logit_scale = (standardized_x0.size(-1) ** 0.5)
                    loss, denoising_loss, _ = self.loss_fn.calculate_total_loss(
                        x0_pred=x0_pred,
                        x0_target=standardized_x0,
                        input_ids=input_ids,
                        embedding_weight=std_embedding_matrix,
                        t_indices=timesteps,
                        alphas_cumprod=self.noise_scheduler.alphas_cumprod,
                        attention_mask=attention_mask,
                        round_threshold=0.15,
                        logit_scale=logit_scale
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / grad_accum_steps
                
                # Safeguard against NaN Loss before backward
                if torch.isnan(loss):
                    print(f"\n[WARNING] NaN loss detected! Skipping batch.")
                    continue
                    
                # 6. Backward
                loss.backward()
                
                # Optimizer Step (Gradient Accumulation)
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item() * grad_accum_steps:.4f}", 
                        "Denoising": f"{denoising_loss.item():.4f}"
                    })
                    
                    # Periodic Preview
                    if global_step % self.config["training"].get("preview_steps", 20) == 0:
                        best_idx = torch.argmin(timesteps).item()
                        self._generate_preview(global_step, x0_pred[best_idx], input_ids[best_idx], timesteps[best_idx].item())
                    
                    # Save checkpoint occasionally
                    if global_step % self.config["training"].get("save_steps", 500) == 0:
                        self._save_checkpoint(global_step)
                    
                    if max_steps is not None and global_step >= max_steps:
                        break
            
            if max_steps is not None and global_step >= max_steps:
                break
                
        progress_bar.close()
        if max_steps is not None and global_step >= max_steps:
            print(f"\n⚡ Reached max_steps ({max_steps}). Stopping training.")

    def _collate_fn(self, features):
        """
        Pad sequences to the same length in the batch.
        """
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        # Pad to max length in this batch (or max_seq_length if defined)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
        
    def _generate_preview(self, step, single_x0_pred, single_target_ids, t_value):
        """
        Decodes a single sample prediction in standardized space.
        """
        with torch.no_grad():
            self.model.eval()
            embedding_matrix = self.model.get_input_embeddings().weight
            std_embedding_matrix = embedding_matrix
            
            # logit_scale for stability
            logit_scale = (single_x0_pred.size(-1) ** 0.5)
            
            # Map latent to logits via standardized embedding matrix
            logits = torch.matmul(single_x0_pred.to(std_embedding_matrix.dtype), std_embedding_matrix.t()) 
            logits = logits / logit_scale
            
            # Diagnostic stats
            l_min, l_max = logits.min().item(), logits.max().item()
            
            pred_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            clean_text = self.tokenizer.decode(single_target_ids, skip_special_tokens=True)
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            print(f"\n--- PREVIEW (Step {step} | t={t_value}) ---")
            print(f"Logit Range: [{l_min:.2f}, {l_max:.2f}]")
            print(f"Target: {clean_text[:200]}...")
            print(f"Predict: {pred_text[:200]}...")
            print("-" * 38 + "\n")
            
            self.model.train()

    def _save_checkpoint(self, step):
        output_dir = self.config["training"].get("output_dir", "./thunder_diffusion_checkpoints")
        path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        
        # We are likely using PEFT (LoRA)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        
        # Save custom heads
        adapter = PrefixLMDiffusionAdapter(self.model)
        adapter.save_diffusion_layers(path)
        print(f"\n⚡ Checkpoint saved to {path}")
        
        # Keep only the last 'save_total_limit' checkpoints
        self.saved_checkpoints.append(path)
        save_total_limit = self.config["training"].get("save_total_limit", 3)
        if save_total_limit is not None and len(self.saved_checkpoints) > save_total_limit:
            import shutil
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                shutil.rmtree(old_checkpoint)
                print(f"⚡ Removed old checkpoint {old_checkpoint}")

def run_training():
    print("⚡ Thunder Trace: Initializing run_training...")
    loader = ThunderModelLoader()
    # Load LLaDA
    print(f"⚡ Thunder Trace: Calling loader.load_model(inference_mode=False) for {loader.model_name}...")
    model, tokenizer = loader.load_model(load_in_4bit=THUNDER_CONFIG["hardware"]["load_in_4bit"], inference_mode=False)
    print("⚡ Thunder Trace: Model and tokenizer loaded.")
    
    # 1. Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Adapt the model for PrefixLM Diffusion
    adapter = PrefixLMDiffusionAdapter(model)
    
    # Target all linear layers for profound capability shift
    from peft import PeftModel
    if not isinstance(model, PeftModel):
        print("⚡ Thunder PrefixLM: Applying new LoRA adapters to LLaDA...")
        # LLaDA uses standard attention names
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model = adapter.apply_lora(r=128, lora_alpha=256, target_modules=target_modules) 
    else:
        print("⚡ Thunder PrefixLM: Model already has LoRA adapters. Skipping re-application.")
        
    # The model is already adapted for diffusion inside loader.load_model()
    
    # 3. Load dataset
    print("⚡ Thunder Trace: Preparing datasets...")
    pipeline = ThunderDataPipeline(tokenizer)
    dataset_name = THUNDER_CONFIG["pipeline"].get("dataset_name")
    print(f"⚡ Thunder: Preparing dataset {dataset_name}...")
    dataset = pipeline.prepare_dataset(dataset_name, augment=False)
    print("⚡ Thunder Trace: Dataset prepared.")
    
    # dataset = dataset.select(range(min(5000, len(dataset)))) 
    
    # 4. Train
    # Add diffusion/training specific overrides directly to the global config
    THUNDER_CONFIG["training"]["t_round_penalty"] = 0.05
    THUNDER_CONFIG["training"]["preview_steps"] = 20 # Balanced preview frequency
    
    # Pass the full root config
    trainer = DiffusionLMTrainer(model, tokenizer, THUNDER_CONFIG)
    trainer.train(dataset)

if __name__ == "__main__":
    run_training()

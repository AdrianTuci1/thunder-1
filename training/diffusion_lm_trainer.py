import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import THUNDER_CONFIG
from core.model_loader import ThunderModelLoader
from core.diffusion_model import PrefixLMDiffusionAdapter
from training.noise_scheduler import ThunderNoiseScheduler
from training.loss_functions import DiffusionLMLoss
from training.data_pipeline import ThunderDataPipeline

class DiffusionLMTrainer:
    """
    Custom training loop for continuous Diffusion-LM with Llama-3.2 (PrefixLM).
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
        self.loss_fn = DiffusionLMLoss(t_round_penalty=config.get("t_round_penalty", 0.05))
        
        self.device = self.model.device
        self.dtype = self.model.dtype

    def train(self, dataset):
        """
        Main training loop.
        """
        print(f"⚡ Thunder PrefixLM: Starting custom training loop on {self.device}...")
        
        # Prepare DataLoader
        batch_size = self.config.get("batch_size", 4)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn
        )
        
        # Optimizer
        learning_rate = self.config.get("learning_rate", 5e-5)
        # We want to optimize the base model (adapters) and our custom heads (x0_head, timestep_embedder)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        epochs = self.config.get("epochs", 3)
        num_training_steps = epochs * len(dataloader)
        
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )
        
        self.model.train()
        
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            progress_bar = tqdm(total=len(dataloader), desc="Training")
            
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # 1. Get exact clean embeddings (x0) and STANDARDIZE them
                # Llama-3.2 embeddings have std ~0.0197, mean ~0.0
                emb_std = 0.02 # Calculated from diagnostics
                embedding_matrix = self.model.get_input_embeddings().weight # [V, H]
                
                # Standardized Matrix for Rounding Loss
                std_embedding_matrix = embedding_matrix / emb_std
                
                clean_embeddings = self.model.get_input_embeddings()(input_ids) # [B, L, H]
                # Map $x_0$ to $N(0, 1)$ space
                standardized_x0 = clean_embeddings / emb_std
                
                # 2. Sample random timesteps
                bsz = input_ids.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_train_timesteps, (bsz,), 
                    device=self.device
                ).long()
                
                # 3. Add noise in the Standardized Space ($N(0, 1)$)
                noise = torch.randn_like(standardized_x0)
                # The scheduler works with cumprod alphas [0, 1], perfect for $N(0, 1)$ space
                noisy_latents = self.noise_scheduler.add_noise(standardized_x0, noise, timesteps)
                
                # Default device type for autocast
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'

                # Use Automatic Mixed Precision for Forward and Loss
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    # 4. Forward Pass (PrefixLM)
                    if not hasattr(self.model, "diffusion_forward"):
                        raise RuntimeError("Model must be adapted for diffusion first.")
                    
                    x0_pred = self.model.diffusion_forward(
                        x_t=noisy_latents, 
                        t=timesteps, 
                        attention_mask=attention_mask
                    )
                    
                    # 5. Compute Diffusion Losses (in Standardized Space)
                    logit_scale = (standardized_x0.size(-1) ** 0.5) + 1e-6
                    
                    loss, mse_loss, l_round_loss = self.loss_fn.calculate_total_loss(
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
                
                # Safeguard against NaN Loss before backward
                if torch.isnan(loss):
                    print(f"\n[WARNING] NaN loss detected at step {global_step}! Skipping step.")
                    optimizer.zero_grad()
                    continue
                    
                # 6. Backward & Step
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "MSE": f"{mse_loss.item():.4f}", 
                    "RND": f"{l_round_loss.item():.4f}" if l_round_loss > 0 else "N/A"
                })
                global_step += 1
                
                # Periodic Preview: Decode x0 for the sample with the LOWEST noise in this batch
                # to see if coherence is emerging where it should.
                if global_step % self.config.get("preview_steps", 20) == 0:
                    # Find index of minimum t in the batch
                    best_idx = torch.argmin(timesteps).item()
                    self._generate_preview(x0_pred[best_idx], input_ids[best_idx], timesteps[best_idx].item())
                
                # Save checkpoint occasionally
                if global_step % self.config.get("save_steps", 500) == 0:
                    self._save_checkpoint(global_step)
                    
            progress_bar.close()

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
        
    def _generate_preview(self, single_x0_pred, single_target_ids, t_value):
        """
        Decodes a single sample prediction in standardized space.
        """
        with torch.no_grad():
            # Standardize logic for preview
            emb_std = 0.02
            embedding_matrix = self.model.get_input_embeddings().weight.detach()
            std_embedding_matrix = embedding_matrix / emb_std
            
            # logit_scale for stability
            logit_scale = (single_x0_pred.size(-1) ** 0.5)
            
            # Map latent to logits via standardized embedding matrix
            # x0_pred is N(0, 1)
            logits = torch.matmul(single_x0_pred.to(std_embedding_matrix.dtype), std_embedding_matrix.t()) 
            logits = logits / logit_scale
            
            pred_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            clean_text = self.tokenizer.decode(single_target_ids, skip_special_tokens=True)
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            print(f"\n--- PREVIEW (Step | t={t_value}) ---")
            print(f"Target:  {clean_text[:120]}...")
            print(f"Predict: {pred_text[:120]}...")
            print(f"--------------------------------------\n")

    def _save_checkpoint(self, step):
        output_dir = self.config.get("output_dir", "./thunder_diffusion_checkpoints")
        path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        
        # We are likely using PEFT (LoRA)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        
        # Save custom heads
        adapter = PrefixLMDiffusionAdapter(self.model)
        adapter.save_diffusion_layers(path)
        print(f"\n⚡ Checkpoint saved to {path}")

def run_training():
    loader = ThunderModelLoader()
    # Load base Llama 3.2 3B
    model, tokenizer = loader.load_model(load_in_4bit=THUNDER_CONFIG["hardware"]["load_in_4bit"])
    
    # 1. Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Adapt the model for PrefixLM Diffusion
    adapter = PrefixLMDiffusionAdapter(model)
    
    # Target all linear layers for profound capability shift (from causal to bidirectional)
    from peft import PeftModel
    if not isinstance(model, PeftModel):
        print("⚡ Thunder PrefixLM: Applying new LoRA adapters...")
        model = adapter.apply_lora(r=128, lora_alpha=256) 
    else:
        print("⚡ Thunder PrefixLM: Model already has LoRA adapters. Skipping re-application.")
        
    model = adapter.adapt_for_diffusion()
    
    # 3. Load dataset
    pipeline = ThunderDataPipeline(tokenizer)
    dataset_name = THUNDER_CONFIG["training"].get("dataset_name", "Open-Orca/OpenOrca")
    print(f"⚡ Thunder: Preparing dataset {dataset_name}...")
    dataset = pipeline.prepare_dataset(dataset_name, augment=False)
    
    # Optional: grab a tiny subset for sanity checking if testing
    dataset = dataset.select(range(min(5000, len(dataset)))) 
    
    # 4. Train
    trainer_config = THUNDER_CONFIG["training"]
    # Add diffusion specific config
    trainer_config["t_round_penalty"] = 0.05
    trainer_config["epochs"] = 3
    trainer_config["preview_steps"] = 20 # Balanced preview frequency
    
    trainer = DiffusionLMTrainer(model, tokenizer, trainer_config)
    trainer.train(dataset)

if __name__ == "__main__":
    run_training()

import os
import sys
import json
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from peft import PeftModel
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
    Custom training loop for continuous Diffusion-LM with Qwen3.5-9B (PrefixLM).
    We do NOT use SFTTrainer because we need tight control over:
    1. The continuous embedding bridge.
    2. The diffusion timestep sampling.
    3. The x0-parametrization MSE loss and L_round penalty.
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.root_config = config if "training" in config else {
            "training": config,
            "hardware": THUNDER_CONFIG.get("hardware", {}),
            "pipeline": THUNDER_CONFIG.get("pipeline", {}),
            "diffusion": THUNDER_CONFIG.get("diffusion", {}),
        }
        self.training_config = self.root_config.get("training", {})
        self.hardware_config = {**THUNDER_CONFIG.get("hardware", {}), **self.root_config.get("hardware", {})}
        self.pipeline_config = {**THUNDER_CONFIG.get("pipeline", {}), **self.root_config.get("pipeline", {})}
        self.diffusion_config = {**THUNDER_CONFIG.get("diffusion", {}), **self.root_config.get("diffusion", {})}
        
        self.noise_scheduler = ThunderNoiseScheduler()
        self.loss_fn = DiffusionLMLoss(t_round_penalty=self.training_config.get("t_round_penalty", 0.05))
        
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.output_dir = self.training_config.get("output_dir", "./thunder_diffusion_checkpoints")
        self.metrics_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.global_step = 0
        self.start_epoch = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, dataset):
        """
        Main training loop.
        """
        print(f"⚡ Thunder PrefixLM: Starting custom training loop on {self.device}...")
        
        # Prepare DataLoader
        # Pull batching from hardware config if available, fallback to training config
        batch_size = self.hardware_config.get("batch_size") or self.training_config.get("batch_size", 4)
        grad_accum = self.hardware_config.get("grad_accum") or self.training_config.get("grad_accum", 1)
        num_workers = self.pipeline_config.get("num_proc", 0)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory="cuda" in str(self.device),
        )
        
        # Optimizer
        learning_rate = self.training_config.get("learning_rate", 5e-5)
        # We want to optimize the base model (adapters) and our custom heads (x0_head, timestep_embedder)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        epochs = self.training_config.get("epochs", 3)
        num_training_steps = epochs * len(dataloader)
        
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.training_config.get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )

        resume_from = self.training_config.get("resume_from")
        if resume_from:
            self._load_training_state(resume_from, optimizer, lr_scheduler)
        
        self.model.train()

        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            progress_bar = tqdm(total=len(dataloader), desc="Training")
            
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # 1. Get exact clean embeddings (x0)
                # We no longer force manual scaling (emb_std), allowing end-to-end training
                # to construct its own semantic space geometry directly.
                embedding_matrix = self.model.get_input_embeddings().weight # [V, H]
                
                # Use raw Matrix for Rounding Loss
                std_embedding_matrix = embedding_matrix
                
                clean_embeddings = self.model.get_input_embeddings()(input_ids) # [B, L, H]
                # Map $x_0$ to diffusion space using raw embeddings
                standardized_x0 = clean_embeddings
                
                # 2. Sample random timesteps
                bsz = input_ids.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.diffusion_steps, (bsz,), 
                    device=self.device
                ).long()
                
                # 3. Add noise in the Standardized Space ($N(0, 1)$)
                noise = torch.randn_like(standardized_x0)
                # The scheduler works with cumprod alphas [0, 1], perfect for $N(0, 1)$ space
                noisy_latents = self.noise_scheduler.add_noise(standardized_x0, noise, timesteps)
                
                # Default device type for autocast
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                autocast_context = (
                    torch.autocast(
                        device_type=device_type,
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    )
                    if device_type == "cuda"
                    else nullcontext()
                )

                # Use Automatic Mixed Precision for Forward and Loss
                with autocast_context:
                    # 4. Forward Pass (PrefixLM)
                    if not hasattr(self.model, "diffusion_forward"):
                        raise RuntimeError("Model must be adapted for diffusion first.")
                    
                    # --- Target: Classifier-Free Guidance and Self-Conditioning ---
                    
                    # A. Prompt Dropout (CFG Training)
                    # We drop the conditional information (attention mask) 15% of the time
                    # to train the unconditional prior for CFG at inference.
                    cfg_mask = attention_mask.clone()
                    # Prompt Dropout for Classifier-Free Guidance Training (CFG)
                    cfg_drop_rate = self.config["diffusion"].get("cfg_drop_rate", 0.15)
                    if torch.rand(1).item() < cfg_drop_rate:
                        # Zeroing mask simulates an unconditional / padding-only input sequence
                        cfg_mask = torch.zeros_like(cfg_mask)

                    # B. Dual-Pass Self-Conditioning
                    # 50% of the time, we do a no-grad pass to guess x0, then condition on it
                    self_cond = None
                    if torch.rand(1).item() < 0.5:
                        with torch.no_grad():
                            self_cond = self.model.diffusion_forward(
                                x_t=noisy_latents, 
                                t=timesteps, 
                                attention_mask=cfg_mask
                            ).detach()
                    
                    # Ensure we are using a non-causal mask explicitly in trainer too
                    # By passing a mask of ones, we override any default causal behavior
                    # if the model was correctly adapted.
                    x0_pred = self.model.diffusion_forward(
                        x_t=noisy_latents, 
                        t=timesteps, 
                        attention_mask=cfg_mask,
                        self_cond=self_cond
                    )
                    
                    # 5. Compute Diffusion Losses (in Standardized Space)
                    logit_scale = (standardized_x0.size(-1) ** 0.5) + 1e-6
                    
                    loss, denoising_loss, _ = self.loss_fn.calculate_total_loss(
                        x0_pred=x0_pred,
                        x0_target=standardized_x0,
                        input_ids=input_ids,
                        embedding_weight=std_embedding_matrix,
                        t_indices=timesteps,
                        alphas_cumprod=self.noise_scheduler.alphas_cumprod,
                        attention_mask=attention_mask, # Loss is calculated only on valid (non-padded) tokens
                        round_threshold=0.15,
                        logit_scale=logit_scale
                    )
                
                # Safeguard against NaN Loss before backward
                if torch.isnan(loss):
                    print(f"\n[WARNING] NaN loss detected at step {self.global_step}! Skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
                # 6. Backward & Step (with Gradient Accumulation)
                # Scale loss by accumulation steps
                scaled_loss = loss / grad_accum
                scaled_loss.backward()
                
                loss_value = loss.detach().float().item()
                denoising_loss_value = denoising_loss.detach().float().item()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    lr_value = lr_scheduler.get_last_lr()[0]
                    self._log_metrics(
                        {
                            "timestamp": time.time(),
                            "epoch": epoch + 1,
                            "step": self.global_step,
                            "loss": loss_value,
                            "denoising_loss": denoising_loss_value,
                            "avg_timestep": float(timesteps.float().mean().item()),
                            "learning_rate": lr_value,
                            "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                            "batch_size": int(input_ids.shape[0]),
                            "sequence_length": int(input_ids.shape[1]),
                            "tokens": int(input_ids.numel()),
                        }
                    )
                    progress_bar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr_value:.2e}")
                
                # Periodic Preview: Decode x0 for the sample with the LOWEST noise in this batch
                # to see if coherence is emerging where it should.
                preview_steps = self.training_config.get("preview_steps", 20)
                if self.global_step > 0 and self.global_step % preview_steps == 0:
                    # Find index of minimum t in the batch
                    best_idx = torch.argmin(timesteps).item()
                    self._generate_preview(x0_pred[best_idx], input_ids[best_idx], timesteps[best_idx].item())
                
                # Save checkpoint occasionally
                save_steps = self.training_config.get("save_steps", 500)
                if self.global_step > 0 and self.global_step % save_steps == 0:
                    self._save_checkpoint(self.global_step, optimizer, lr_scheduler, epoch)

                progress_bar.update(1)
                    
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
            # Raw embeddings for preview decoding
            embedding_matrix = self.model.get_input_embeddings().weight.detach()
            std_embedding_matrix = embedding_matrix
            
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

    def _save_checkpoint(self, step, optimizer, lr_scheduler, epoch):
        path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        
        # We are likely using PEFT (LoRA)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        
        # Save custom heads
        adapter = PrefixLMDiffusionAdapter(self.model)
        adapter.save_diffusion_layers(path)
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        trainer_state = {
            "global_step": step,
            "epoch": epoch,
            "metrics_path": self.metrics_path,
        }
        with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as handle:
            json.dump(trainer_state, handle, indent=2)
        print(f"\n⚡ Checkpoint saved to {path}")

    def _load_training_state(self, checkpoint_path, optimizer, lr_scheduler):
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        state_path = os.path.join(checkpoint_path, "trainer_state.json")

        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        if os.path.exists(scheduler_path):
            lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as handle:
                trainer_state = json.load(handle)
            self.global_step = int(trainer_state.get("global_step", 0))
            self.start_epoch = int(trainer_state.get("epoch", 0))

    def _log_metrics(self, payload):
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

def run_training():
    loader = ThunderModelLoader()
    # Load base Qwen3.5-9B
    model, tokenizer = loader.load_model(load_in_4bit=THUNDER_CONFIG["hardware"]["load_in_4bit"])
    
    # 1. Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Adapt the model for PrefixLM Diffusion
    adapter = PrefixLMDiffusionAdapter(model)
    
    # Target all linear layers for profound capability shift (from causal to bidirectional)
    if not isinstance(model, PeftModel):
        print("⚡ Thunder PrefixLM: Applying new LoRA adapters...")
        model = adapter.apply_lora(
            r=THUNDER_CONFIG["training"]["lora_rank"], 
            lora_alpha=THUNDER_CONFIG["training"]["lora_alpha"]
        ) 
    else:
        print("⚡ Thunder PrefixLM: Model already has LoRA adapters. Skipping re-application.")
        
    model = adapter.adapt_for_diffusion()
    
    # 3. Load dataset
    pipeline = ThunderDataPipeline(tokenizer)
    dataset_names = THUNDER_CONFIG["pipeline"]["dataset_name"]
    print(f"⚡ Thunder: Preparing dataset mix...")
    dataset = pipeline.prepare_dataset(dataset_names, augment=True) # Always augment (noise) for diffusion training
    
    # Optional: grab a tiny subset for sanity checking if testing
    dataset = dataset.select(range(min(5000, len(dataset)))) 
    
    # 4. Train
    trainer_config = {
        **THUNDER_CONFIG,
        "training": {
            **THUNDER_CONFIG["training"],
            "t_round_penalty": 0.05,
            "epochs": 3,
            "preview_steps": 20,
        },
    }
    
    trainer = DiffusionLMTrainer(model, tokenizer, trainer_config)
    trainer.train(dataset)

if __name__ == "__main__":
    run_training()

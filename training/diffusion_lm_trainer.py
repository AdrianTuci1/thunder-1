import os
import sys
import json
import time
from typing import Optional, Iterable, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import THUNDER_CONFIG
from training.noise_scheduler import ThunderNoiseScheduler
from training.loss_functions import DiffusionLMLoss

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
from core.model_loader import ThunderModelLoader
from training.data_pipeline import ThunderDataPipeline
from core.storage import ObjectStorageManager

class DiffusionLMTrainer:
    """
    Robust training loop for Thunder Diffusion-LM, scaling to Multi-GPU via Accelerator.
    Optimized for 'The Big Run' (GQA, 8k Context, Romanian datasets).
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.root_config = config
        self.training_config = config.get("training", {})
        self.hardware_config = config.get("hardware", {})
        self.pipeline_config = config.get("pipeline", {})
        self.diffusion_config = config.get("diffusion", {})
        
        self.noise_scheduler = ThunderNoiseScheduler()
        self.loss_fn = DiffusionLMLoss(t_round_penalty=self.training_config.get("t_round_penalty", 0.0))
        
        # Initialize Accelerator for Multi-GPU and Mixed Precision
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.hardware_config.get("grad_accum", 1),
            log_with="wandb" if self.training_config.get("use_wandb", False) else None,
        )
        
        self.device = self.accelerator.device
        self.output_dir = self.training_config.get("output_dir", "./thunder_checkpoints")
        self.global_step = 0
        self.start_epoch = 0
        
        # Curriculum settings
        self.curriculum_lengths = self.pipeline_config.get("curriculum_lengths", [8192])
        self.curriculum_stage_steps = self.training_config.get("curriculum_stage_steps", 5000)
        
        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

        # Init tracking (WandB)
        if self.training_config.get("use_wandb", False) and wandb is not None:
            self.accelerator.init_trackers(
                project_name=self.training_config.get("wandb_project", "thunder-dllm"),
                config=self.root_config,
                init_kwargs={
                    "wandb": {
                        "name": self.training_config.get("wandb_run_name", f"run-{int(time.time())}"),
                    }
                }
            )
        
        # Initialize storage manager for R2/S3 syncing
        self.storage_manager = ObjectStorageManager(config)

    def train(self, dataset):
        batch_size = self.hardware_config.get("batch_size", 2)
        epochs = self.training_config.get("epochs", 1)
        is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=not is_iterable, 
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 2e-4),
            weight_decay=self.training_config.get("weight_decay", 0.1),
        )
        
        # Estimate training steps
        if is_iterable:
            steps_per_epoch = self.training_config.get("max_train_blocks", 1000000) // batch_size
        else:
            steps_per_epoch = len(dataloader)
        
        num_training_steps = epochs * steps_per_epoch
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.training_config.get("warmup_steps", 2000),
            num_training_steps=num_training_steps
        )

        # Accelerator Prepare (Multi-GPU setup)
        self.model, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, dataloader, lr_scheduler
        )

        # Resume logic
        resume_from = self.training_config.get("resume_from")
        if resume_from:
            self._load_training_state(resume_from, optimizer, lr_scheduler)
        
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, epochs):
            if self.accelerator.is_main_process:
                print(f"\n🚀 Epoch {epoch+1}/{epochs}")
            
            progress_bar = tqdm(
                total=steps_per_epoch, 
                desc="Training", 
                disable=not self.accelerator.is_main_process
            )
            
            for step, batch in enumerate(dataloader):
                if is_iterable and step >= steps_per_epoch:
                    break
                    
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                input_ids, attention_mask = self._apply_length_curriculum(input_ids, attention_mask)
                
                with self.accelerator.accumulate(self.model):
                    # 1. Forward Pass logic
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    embedding_matrix = unwrapped.get_input_embeddings().weight
                    clean_embeddings = unwrapped.get_input_embeddings()(input_ids)
                    
                    bsz = input_ids.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.diffusion_steps, (bsz,), device=self.device).long()
                    noise = torch.randn_like(clean_embeddings)
                    noisy_latents = self.noise_scheduler.add_noise(clean_embeddings, noise, timesteps)
                    
                    # CFG Training (Prompt Dropout)
                    cfg_mask = attention_mask.clone()
                    if torch.rand(1).item() < self.diffusion_config.get("cfg_drop_rate", 0.1):
                        cfg_mask = torch.zeros_like(cfg_mask)

                    # Self-Conditioning (Pass 1)
                    self_cond = None
                    if self.training_config.get("self_conditioning", True) and torch.rand(1).item() < 0.5:
                        with torch.no_grad():
                            if HAS_TE and self.training_config.get("use_fp8", False):
                                import transformer_engine.pytorch as te
                                from transformer_engine.common.recipe import Format, DelayedScaling
                                fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=16, amax_compute_algo="max")
                                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                                    self_cond = self.model.diffusion_forward(noisy_latents, timesteps, cfg_mask).detach()
                            else:
                                self_cond = self.model.diffusion_forward(noisy_latents, timesteps, cfg_mask).detach()
                    
                    # Pass 2: Final Prediction
                    if HAS_TE and self.training_config.get("use_fp8", False):
                        import transformer_engine.pytorch as te
                        from transformer_engine.common.recipe import Format, DelayedScaling
                        fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=16, amax_compute_algo="max")
                        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                            x0_pred = self.model.diffusion_forward(noisy_latents, timesteps, cfg_mask, self_cond=self_cond)
                    else:
                        x0_pred = self.model.diffusion_forward(noisy_latents, timesteps, cfg_mask, self_cond=self_cond)
                    
                    # 2. Loss Calculation
                    loss, denoising_loss, _ = self.loss_fn.calculate_total_loss(
                        x0_pred=x0_pred,
                        x0_target=clean_embeddings,
                        input_ids=input_ids,
                        embedding_weight=embedding_matrix,
                        t_indices=timesteps,
                        alphas_cumprod=self.noise_scheduler.alphas_cumprod,
                        attention_mask=attention_mask
                    )
                    
                    if not torch.isfinite(loss):
                        self.accelerator.print(f"⚠️ [WARNING] NaN loss at step {self.global_step}. Skipping.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # 3. Backward & Step
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        progress_bar.update(1)
                        lr_val = lr_scheduler.get_last_lr()[0]
                        metrics = {
                            "loss": loss.item(),
                            "denoising_loss": denoising_loss.item(),
                            "lr": lr_val,
                            "step": self.global_step
                        }
                        self.accelerator.log(metrics, step=self.global_step)
                        progress_bar.set_postfix({"loss": f"{metrics['loss']:.4f}", "lr": f"{lr_val:.2e}"})

                        # Periodically save checkpoints
                        if self.global_step % self.training_config.get("save_steps", 500) == 0:
                            self._save_checkpoint(self.global_step, optimizer, lr_scheduler, epoch)

                        # Periodically generate previews
                        if self.global_step % self.training_config.get("preview_steps", 100) == 0:
                            self._generate_preview(input_ids[0], x0_pred[0], timesteps[0].item())

            progress_bar.close()

    def _save_checkpoint(self, step, optimizer, lr_scheduler, epoch):
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), os.path.join(ckpt_path, "model_state.pt"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
        
        metadata = {"step": step, "epoch": epoch, "config": self.root_config}
        with open(os.path.join(ckpt_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"💾 Checkpoint saved: {ckpt_path}")
        
        # Sync to Object Storage (R2/S3) if enabled
        self.storage_manager.upload_checkpoint_async(ckpt_path)

    def _load_training_state(self, path, optimizer, lr_scheduler):
        print(f"🔄 Resuming from {path}...")
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(torch.load(os.path.join(path, "model_state.pt"), map_location=self.device, weights_only=True))
        
        if os.path.exists(os.path.join(path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt"), map_location=self.device))
        if os.path.exists(os.path.join(path, "scheduler.pt")):
            lr_scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt"), map_location=self.device))
        if os.path.exists(os.path.join(path, "metadata.json")):
            with open(os.path.join(path, "metadata.json"), "r") as f:
                meta = json.load(f)
                self.global_step = meta.get("step", 0)
                self.start_epoch = meta.get("epoch", 0)

    def _generate_preview(self, input_ids, x0_pred, t_idx):
        with torch.no_grad():
            unwrapped = self.accelerator.unwrap_model(self.model)
            embedding_matrix = unwrapped.get_input_embeddings().weight
            logits = torch.matmul(x0_pred.float(), embedding_matrix.float().t())
            tokens = torch.argmax(logits, dim=-1)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            target = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"\n--- 🔍 PREVIEW (Step {self.global_step} | t={t_idx}) ---")
            print(f"Target:   {target[:120]}...")
            print(f"Predict:  {text[:120]}...")
            print("-" * 50)

    def _apply_length_curriculum(self, input_ids, attention_mask):
        stage_idx = min(self.global_step // self.curriculum_stage_steps, len(self.curriculum_lengths) - 1)
        target_len = self.curriculum_lengths[stage_idx]
        if input_ids.shape[1] > target_len:
            start = torch.randint(0, input_ids.shape[1] - target_len + 1, (1,)).item()
            return input_ids[:, start:start+target_len], attention_mask[:, start:start+target_len]
        return input_ids, attention_mask

    def _collate_fn(self, features):
        ids = [torch.tensor(f["input_ids"]) for f in features]
        padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return {"input_ids": padded, "attention_mask": (padded != self.tokenizer.pad_token_id).long()}

def run_training():
    loader = ThunderModelLoader()
    model, tokenizer = loader.load_model(load_in_4bit=THUNDER_CONFIG["hardware"]["load_in_4bit"])
    
    pipeline = ThunderDataPipeline(tokenizer)
    dataset = pipeline.prepare_dataset()
    
    trainer = DiffusionLMTrainer(model, tokenizer, THUNDER_CONFIG)
    trainer.train(dataset)

if __name__ == "__main__":
    run_training()

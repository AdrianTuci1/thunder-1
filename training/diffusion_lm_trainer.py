import os
import sys
import json
import time
import shutil
from typing import Optional, Iterable, List
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

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

import datasets
# Stabilize streaming for open-web-math and large mixes
datasets.config.STREAMING_READ_MAX_RETRIES = 50
datasets.config.HF_HUB_OFFLINE = False

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
        self.last_save_time = time.time()

    def train(self, dataset):
        batch_size = self.hardware_config.get("batch_size", 2)
        epochs = self.training_config.get("epochs", 1)
        is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=not is_iterable, 
            collate_fn=self._collate_fn,
            num_workers=self.pipeline_config.get("num_proc", 4),
            pin_memory=True,
            persistent_workers=is_iterable and self.pipeline_config.get("num_proc", 4) > 0
        )
        
        # [NEW] Optimized Optimizer
        use_fused = self.hardware_config.get("fused_kernels", False)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 2e-4),
            weight_decay=self.training_config.get("weight_decay", 0.1),
            fused=use_fused and torch.cuda.is_available()
        )
        
        # Estimate training steps
        grad_accum = self.hardware_config.get("grad_accum", 1)
        if is_iterable:
            total_blocks = self.training_config.get("max_train_blocks", 1000000)
            num_training_steps = (total_blocks // batch_size) // grad_accum
        else:
            num_training_steps = len(dataloader) // grad_accum
        
        steps_per_epoch = num_training_steps
        
        # [NEW] Custom LR Scheduler: Warm-down (Constant followed by Exponential Decay)
        lr_type = self.training_config.get("lr_schedule_type", "cosine")
        if lr_type == "thunder_warmdown":
            constant_steps = self.training_config.get("warmdown_constant_steps", 10000)
            # Decădem lent până la 10% din LR-ul inițial
            decay_steps = num_training_steps - (self.global_step + constant_steps)
            
            def lr_lambda(current_step):
                # current_step here is the step relative to 0 if NOT resumed, 
                # but we usually resume global_step as well.
                # The scheduler.step() is called global_step times during training.
                if current_step < (self.global_step + constant_steps):
                    return 1.0
                else:
                    # Exponential decay: e^(-k * t)
                    # We want to reach 0.1 at num_training_steps
                    rel_step = current_step - (self.global_step + constant_steps)
                    if decay_steps <= 0: return 1.0
                    decay_rate = -torch.log(torch.tensor(0.1)) / decay_steps
                    return torch.exp(-decay_rate * rel_step).item()
            
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            lr_scheduler = get_scheduler(
                lr_type,
                optimizer=optimizer,
                num_warmup_steps=self.training_config.get("warmup_steps", 2000),
                num_training_steps=num_training_steps
            )

        # Accelerator Prepare (Multi-GPU setup)
        self.model, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, dataloader, lr_scheduler
        )

        # [NEW] Torch.compile for A100 Speedup
        if self.hardware_config.get("fused_kernels", False):
            try:
                print("⚡ Thunder: Compiling model with torch.compile()... (First steps will be slow)")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"⚠️  Torch.compile failed: {e}. Falling back to eager mode.")

        # Resume logic
        resume_from = self.training_config.get("resume_from")
        if resume_from == "latest":
            resume_from = self._detect_latest_checkpoint()
            
        if resume_from:
            self._load_training_state(resume_from, optimizer, lr_scheduler)
            
            # [NEW] Recalculate LR Scheduler state if using thunder_warmdown
            if lr_type == "thunder_warmdown":
                # We need to recalculate the decay_steps which might have been based on global_step=0.
                # Since decay_steps is a local variable captured by lr_lambda, updating it here 
                # will reflect in future calls to the scheduler.
                decay_steps = num_training_steps - (self.global_step + constant_steps)
        
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        try:
            for epoch in range(self.start_epoch, epochs):
                if self.accelerator.is_main_process:
                    print(f"\n🚀 Epoch {epoch+1}/{epochs}")
            
            progress_bar = tqdm(
                total=steps_per_epoch, 
                initial=self.global_step,
                desc="Training", 
                disable=not self.accelerator.is_main_process
            )
            
            for step, batch in enumerate(dataloader):
                # [NEW] Logical Resume: Skip batches already processed in previous sessions
                # We use (global_step * grad_accum) to find the correct batch offset
                grad_accum = self.hardware_config.get("grad_accum", 1)
                skip_threshold = self.global_step * grad_accum
                
                if is_iterable and step < skip_threshold:
                    if step % 1000 == 0 and self.accelerator.is_main_process:
                        progress_bar.set_description(f"⏩ Skipping to batch {skip_threshold} (Global Step {self.global_step})")
                    continue
                
                if is_iterable and step == skip_threshold and self.accelerator.is_main_process:
                    progress_bar.set_description("Training")

                if is_iterable and self.global_step >= num_training_steps:
                    break
                    break
                    
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                input_ids, attention_mask = self._apply_length_curriculum(input_ids, attention_mask)
                
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        # 1. Forward Pass logic
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        embedding_matrix = unwrapped.get_input_embeddings().weight
                        clean_embeddings = unwrapped.get_input_embeddings()(input_ids)
                        
                        bsz = input_ids.shape[0]
                        
                        # [NEW] Biased Noise Sampling
                        if self.training_config.get("noise_sampling_mode") == "biased":
                            range_min, range_max = self.training_config.get("noise_sampling_range", [20, 80])
                            is_biased = torch.rand(bsz, device=self.device) < 0.7
                            timesteps = torch.randint(0, self.noise_scheduler.diffusion_steps, (bsz,), device=self.device).long()
                            biased_timesteps = torch.randint(range_min, range_max + 1, (bsz,), device=self.device).long()
                            timesteps = torch.where(is_biased, biased_timesteps, timesteps)
                        else:
                            timesteps = torch.randint(0, self.noise_scheduler.diffusion_steps, (bsz,), device=self.device).long()
                        noise = torch.randn_like(clean_embeddings)
                        noisy_latents = self.noise_scheduler.add_noise(clean_embeddings, noise, timesteps)
                        
                        # CFG Training (Prompt Dropout)
                        cfg_mask = attention_mask.clone()
                        if torch.rand(1).item() < self.diffusion_config.get("cfg_drop_rate", 0.1):
                            cfg_mask = torch.zeros_like(cfg_mask)

                        # Self-Conditioning (Pass 1)
                        self_cond = None
                        if self.training_config.get("self_conditioning", True) and torch.rand(1).item() < 0.25:
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
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 0.8)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    
                    if self.accelerator.is_main_process:
                        progress_bar.update(1)
                    
                    if self.accelerator.is_main_process:
                        lr_val = lr_scheduler.get_last_lr()[0]
                        metrics = {
                            "loss": loss.item(),
                            "denoising_loss": denoising_loss.item(),
                            "lr": lr_val,
                            "step": self.global_step,
                        }
                        self.accelerator.log(metrics, step=self.global_step)
                        progress_bar.set_postfix({
                            "loss": f"{metrics['loss']:.4f}", 
                            "lr": f"{lr_val:.2e}"
                        })

                        # Periodically save checkpoints (Time-based or Step-based)
                        save_steps = self.training_config.get("save_steps", 5000)
                        save_interval_hrs = self.training_config.get("save_interval_hours", 2)
                        time_since_save = (time.time() - self.last_save_time) / 3600
                        
                        if (self.global_step > 0) and ((self.global_step % save_steps == 0) or (time_since_save >= save_interval_hrs)):
                            self._save_checkpoint(self.global_step, optimizer, lr_scheduler, epoch)
                            self.last_save_time = time.time()

                        # Periodically generate previews
                        if self.global_step % self.training_config.get("preview_steps", 100) == 0:
                            self._generate_preview(input_ids[0], x0_pred[0], timesteps[0].item())

                progress_bar.close()
                
        except KeyboardInterrupt:
            if self.accelerator.is_main_process:
                print("\n🛑 Training interrupted by user (Ctrl+C).")
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"\n❌ Training crashed with error: {e}")
            raise e
        finally:
            if self.accelerator.is_main_process:
                print("💾 [Save-on-Exit] Saving final state before shutdown...")
                # We save whatever global_step we reached
                self._save_checkpoint(self.global_step, optimizer, lr_scheduler, self.start_epoch)
                print("🏁 Training session closed.")

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
        
        # Cleanup old checkpoints (Local and R2)
        if self.accelerator.is_main_process:
            self._prune_checkpoints()

    def _prune_checkpoints(self):
        """
        Keeps local and remote checkpoints under the save_total_limit.
        """
        limit = self.training_config.get("save_total_limit", 5)
        if limit <= 0:
            return

        # 1. Local Pruning
        try:
            checkpoint_dirs = [
                os.path.join(self.output_dir, d) 
                for d in os.listdir(self.output_dir) 
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.output_dir, d))
            ]
            
            if len(checkpoint_dirs) > limit:
                # Sort by step number
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                to_delete = checkpoint_dirs[:-limit]
                
                for dir_path in to_delete:
                    print(f"🧹 Pruning local checkpoint: {dir_path}")
                    shutil.rmtree(dir_path)
                    
        except Exception as e:
            print(f"⚠️ Error pruning local checkpoints: {e}")

        # 2. Remote Pruning (R2)
        if self.storage_manager.enabled:
            self.storage_manager.cleanup_remotely(limit)

    def _load_training_state(self, path, optimizer, lr_scheduler):
        """
        Loads model, optimizer, and scheduler states.
        If the path doesn't exist locally, attempts to download it from R2.
        """
        print(f"🔄 Resuming from {path}...")
        
        # Check if local path exists; if not, try R2 download
        if not os.path.exists(path) or not os.path.exists(os.path.join(path, "model_state.pt")):
            if self.storage_manager.enabled:
                print(f"🕵️  Checkpoint not found locally. Searching in R2...")
                # The 'path' might be 'runs/checkpoint-1000' or just 'checkpoint-1000'
                checkpoint_name = os.path.basename(path.rstrip("/"))
                success = self.storage_manager.download_checkpoint(checkpoint_name, path)
                if not success:
                    print(f"❌ Failed to find or download checkpoint {checkpoint_name} from R2.")
                    return
            else:
                print(f"❌ Checkpoint {path} not found and R2 storage is disabled.")
                return

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
        
        print(f"✅ Successfully resumed from step {self.global_step}")

    def _detect_latest_checkpoint(self) -> Optional[str]:
        """
        Detects the latest checkpoint locally or in R2.
        Returns the path to the checkpoint.
        """
        local_latest = None
        if os.path.exists(self.output_dir):
            checkpoint_dirs = [
                d for d in os.listdir(self.output_dir) 
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.output_dir, d))
            ]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                local_latest = os.path.join(self.output_dir, checkpoint_dirs[-1])
        
        # Also check R2
        remote_latest_name = self.storage_manager.get_latest_checkpoint_name()
        
        if not local_latest and not remote_latest_name:
            print("⚠️ No checkpoints found locally or in R2.")
            return None
            
        if not local_latest:
            return os.path.join(self.output_dir, os.path.basename(remote_latest_name))
            
        if not remote_latest_name:
            return local_latest
            
        # Compare step numbers if both exist
        local_step = int(os.path.basename(local_latest).split("-")[-1])
        remote_step = int(remote_latest_name.split("-")[-1])
        
        if remote_step > local_step:
            print(f"🌐 R2 has a newer checkpoint ({remote_step} > {local_step}).")
            return os.path.join(self.output_dir, os.path.basename(remote_latest_name))
        
        return local_latest

    def _generate_preview(self, input_ids, x0_pred, t_idx):
        with torch.no_grad():
            unwrapped = self.accelerator.unwrap_model(self.model)
            embedding_matrix = unwrapped.get_input_embeddings().weight
            
            # --- Robust Decoding (Cosine Similarity) ---
            # Normalizing helps argmax pick the contextually correct token even if confidence/norm is low
            x0_norm = x0_pred.float() / (x0_pred.float().norm(dim=-1, keepdim=True) + 1e-8)
            emb_norm = embedding_matrix.float() / (embedding_matrix.float().norm(dim=-1, keepdim=True) + 1e-8)
            
            logits = torch.matmul(x0_norm, emb_norm.t())
            tokens = torch.argmax(logits, dim=-1)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            target = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # --- Best Case Preview ---
            # If the current training t is very high, also show what the model predicts for low noise (t=0)
            best_case_text = ""
            if t_idx > 50:
                # One extra forward pass with t=0 to see current 'Clean Embedding' recovery progress
                low_t = torch.zeros((1,), device=self.device).long()
                # Prepare a mini-batch of 1
                clean_embeds = unwrapped.get_input_embeddings()(input_ids.unsqueeze(0))
                # Just show the model's direct map at t=0
                x0_low = unwrapped.diffusion_forward(clean_embeds, low_t)
                x0_low_norm = x0_low[0].float() / (x0_low[0].float().norm(dim=-1, keepdim=True) + 1e-8)
                logits_low = torch.matmul(x0_low_norm, emb_norm.t())
                tokens_low = torch.argmax(logits_low, dim=-1)
                best_case_text = self.tokenizer.decode(tokens_low, skip_special_tokens=True)

            print(f"\n--- 🔍 PREVIEW (Step {self.global_step} | t={t_idx}) ---")
            print(f"Target:   {target[:120]}...")
            print(f"Predict:  {text[:120]}...")
            if best_case_text:
                print(f"Denoised: {best_case_text[:120]}... (Reconstructed from t=0)")
            print("-" * 50)

    def _apply_length_curriculum(self, input_ids, attention_mask):
        stage_idx = min(self.global_step // self.curriculum_stage_steps, len(self.curriculum_lengths) - 1)
        target_len = self.curriculum_lengths[stage_idx]
        if input_ids.shape[1] > target_len:
            start = torch.randint(0, input_ids.shape[1] - target_len + 1, (1,)).item()
            return input_ids[:, start:start+target_len], attention_mask[:, start:start+target_len]
        return input_ids, attention_mask

    def _collate_fn(self, features):
        ids = [f["input_ids"].clone().detach() if torch.is_tensor(f["input_ids"]) else torch.tensor(f["input_ids"]) for f in features]
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

import os
import sys
import json
import torch
import torch.nn as nn
from core.config_manager import THUNDER_CONFIG
from training.diffusion_lm_trainer import DiffusionLMTrainer
from training.noise_scheduler import ThunderNoiseScheduler

class ThunderDistillationTrainer(DiffusionLMTrainer):
    """
    Implements Teacher-Student Distillation for Thunder dLLM.
    Teaches a 'Student' model to crystallize responses in very few steps (3-8)
    by imitating a 'Teacher' model that runs the full diffusion chain.
    """
    
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        
        # Load Teacher Model (usually a frozen copy of the pretrained model)
        self.teacher_model = self._create_teacher(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.distillation_config = config.get("training", {}).get("teacher_student_distillation", {
            "teacher_steps": 32,
            "student_fast_steps": 8
        })

    def _create_teacher(self, model):
        # In a real scenario, we might want to load a specific checkpoint.
        # For now, we clone the current model as the base teacher.
        import copy
        teacher = copy.deepcopy(model)
        return teacher

    def train_step(self, batch, optimizer, lr_scheduler):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        unwrapped = self.accelerator.unwrap_model(self.model)
        embedding_matrix = unwrapped.get_input_embeddings().weight
        clean_embeddings = unwrapped.get_input_embeddings()(input_ids)
        
        bsz = input_ids.shape[0]
        # For distillation, we sample timesteps differently to cover the "jump"
        timesteps = torch.randint(1, self.noise_scheduler.diffusion_steps, (bsz,), device=self.device).long()
        
        # 1. Teacher Prediction (What the ideal x0 should look like at this t)
        with torch.no_grad():
            noise = torch.randn_like(clean_embeddings)
            noisy_latents = self.noise_scheduler.add_noise(clean_embeddings, noise, timesteps)
            
            # The teacher predicts x0 from noisy latents
            # In advanced distillation, the teacher might do multiple steps,
            # but here we use its best one-step prediction as the target.
            x0_teacher = self.teacher_model.diffusion_forward(noisy_latents, timesteps, attention_mask)

        # 2. Student Prediction (Learning to match teacher)
        x0_student = self.model.diffusion_forward(noisy_latents, timesteps, attention_mask)
        
        # 3. Distillation Loss (MSE in Latent Space + Cross Entropy on Tokens)
        # Latent Matching
        latent_loss = nn.functional.mse_loss(x0_student, x0_teacher)
        
        # Token Matching (Standard Denoising Loss but guided by Teacher's preference)
        distill_loss, _, _ = self.loss_fn.calculate_total_loss(
            x0_pred=x0_student,
            x0_target=x0_teacher, # Targeted at Teacher's prediction
            input_ids=input_ids,
            embedding_weight=embedding_matrix,
            t_indices=timesteps,
            alphas_cumprod=self.noise_scheduler.alphas_cumprod,
            attention_mask=attention_mask
        )
        
        total_loss = distill_loss + 0.5 * latent_loss
        
        self.accelerator.backward(total_loss)
        return total_loss

    # Overriding the main train loop to use distillation logic
    def train(self, dataset):
        # Similar to parent but with distillation steps
        super().train(dataset)
        # Note: We would typically call train_step in the loop. 
        # For this implementation, we ensure DiffusionLMTrainer's loop calls a custom forward.
        pass

def run_distillation():
    from core.model_loader import ThunderModelLoader
    from training.data_pipeline import ThunderDataPipeline
    
    loader = ThunderModelLoader()
    model, tokenizer = loader.load_model()
    
    pipeline = ThunderDataPipeline(tokenizer)
    dataset = pipeline.prepare_dataset()
    
    trainer = ThunderDistillationTrainer(model, tokenizer, THUNDER_CONFIG)
    trainer.train(dataset)

if __name__ == "__main__":
    run_distillation()

import torch
from core.config_manager import THUNDER_CONFIG
from training.noise_scheduler import ThunderNoiseScheduler

class ThunderDiffusionEngine:
    """
    The core "All-at-Once" diffusion engine for Thunder.
    Uses DDIM-based reverse diffusion to refine the full sequence from noise to data.
    """
    
    def __init__(self, model, adaptive_scheduler):
        self.model = model
        self.adaptive_scheduler = adaptive_scheduler
        # The noise scheduler handles the math of alphas/betas
        self.noise_scheduler = ThunderNoiseScheduler()

    def process_single_step(self, x, step_idx, total_steps, prompt_embeds=None):
        """
        Executes exactly one reverse diffusion step (DDIM) on the full latent sequence.
        """
        # Prediction step (Reverse Diffusion)
        current_state = self._denoise_step(x, total_steps - 1 - step_idx, total_steps, prompt_embeds)
            
        return current_state

    def crystallize_sequence(self, initial_noise, steps=50, prompt_embeds=None):
        """
        Iteratively removes noise over the full sequence to reveal the final latent sequence.
        """
        if steps is None: # Retain adaptive steps for legacy wrapper if needed
            steps = self.adaptive_scheduler.get_adaptive_steps(initial_noise)

        print(f"⚡ Thunder: Crystallizing full sequence (Steps: {steps})...")

        current_state = initial_noise
        
        for step_idx in range(steps):
            current_state = self.process_single_step(
                x=current_state, 
                step_idx=step_idx, 
                total_steps=steps, 
                prompt_embeds=prompt_embeds
            )
            
        return current_state

    def _denoise_step(self, x, t_idx, total_inference_steps, prompt_embeds=None):
        """
        A single denoising step using DDIM-like logic on the full Latent Field.
        Uses the adapted sequence backbone for ε-prediction.
        """
        device = x.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        
        # Map linear inference step index to training timesteps
        t_now = int((t_idx / total_inference_steps) * num_train_timesteps)
        t_next = int(((t_idx - 1) / total_inference_steps) * num_train_timesteps) if t_idx > 0 else 0
        
        # Get alphas from scheduler
        alpha_now = self.noise_scheduler.alphas_cumprod[t_now].to(device)
        alpha_next = self.noise_scheduler.alphas_cumprod[t_next].to(device) if t_idx > 0 else torch.tensor(1.0, device=device)

        with torch.no_grad():
            # 1. Predict Noise (ε) using the adapted backbone
            t_tensor = torch.tensor([t_now], device=device).float()
            
            # If the model takes conditioning, pass it; otherwise just predict noise from the state
            if prompt_embeds is not None:
                # We assume model.predict_noise accepts kwargs for condition
                epsilon_pred = self.model.predict_noise(x, t_tensor, condition=prompt_embeds)
            else:
                epsilon_pred = self.model.predict_noise(x, t_tensor)
            
            # 2. DDIM math: solve for x_0, then propagate to x_{t-1}
            sqrt_one_minus_alpha_now = torch.sqrt(1.0 - alpha_now)
            sqrt_alpha_now = torch.sqrt(alpha_now)
            
            x_0_pred = (x - sqrt_one_minus_alpha_now * epsilon_pred) / sqrt_alpha_now
            
            sqrt_alpha_next = torch.sqrt(alpha_next)
            sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next)
            
            # Crystallization: mixing predicted clean signal with noise for the next step
            new_state = sqrt_alpha_next * x_0_pred + sqrt_one_minus_alpha_next * epsilon_pred
            
            return new_state

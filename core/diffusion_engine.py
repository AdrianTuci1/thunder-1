import torch
from core.config_manager import THUNDER_CONFIG
from core.boundary_fuser import BoundaryFuser
from training.noise_scheduler import ThunderNoiseScheduler

class ThunderDiffusionEngine:
    """
    The core "All-at-Once" crystallization engine.
    Uses DDIM-based reverse diffusion to refine tiles from noise to data.
    """
    
    def __init__(self, model, adaptive_scheduler):
        self.model = model
        self.adaptive_scheduler = adaptive_scheduler
        self.fuser = BoundaryFuser()
        # The noise scheduler handles the math of alphas/betas
        self.noise_scheduler = ThunderNoiseScheduler()

    def process_single_step(self, x, step_idx, total_steps, macro_context=None, anchor_data=None):
        """
        Executes exactly one reverse diffusion step (DDIM) on a micro-tile's latent field.
        This allows the Orchestrator to synchronize overlaps across all tiles globally per step.
        """
        # 1. Prediction step (Reverse Diffusion)
        current_state = self._denoise_step(x, total_steps - 1 - step_idx, total_steps)
        
        # 2. Apply Global Coherence (Latent Nudge)
        current_state = self.fuser.apply_global_coherence(current_state, macro_context)
        
        # 3. Apply Anchored Fusion (Boundary Constraint from neighboring tile)
        if anchor_data is not None:
            current_state = self._apply_anchor(current_state, anchor_data, step_idx, total_steps)
            
        return current_state

    def crystallize_tile(self, initial_noise, steps=50, macro_context=None, anchor_data=None):
        """
        Legacy/Standalone loop: Iteratively removes noise to reveal the final latent sequence.
        (Used mainly for testing or single-tile generation without global orchestration).
        """
        if steps is None: # Retain adaptive steps for legacy wrapper if needed
            steps = self.adaptive_scheduler.get_adaptive_steps(initial_noise)

        print(f"⚡ Thunder: Crystallizing tile (Steps: {steps}, Anchored: {anchor_data is not None})...")

        current_state = initial_noise
        
        for step_idx in range(steps):
            current_state = self.process_single_step(
                x=current_state, 
                step_idx=step_idx, 
                total_steps=steps, 
                macro_context=macro_context, 
                anchor_data=anchor_data
            )
            
        return current_state

    def _apply_anchor(self, x, anchor_data, step_idx, total_steps):
        """
        Forces the beginning of the current tile to match the end of the 
        previous tile (anchor) within the overlap region.
        Uses refined temporal fusion from BoundaryFuser.
        """
        overlap_size = self.fuser.overlap_size
        
        # Use refined fusion logic that accounts for diffusion steps
        fused_overlap = self.fuser.fuse_anchored_overlap(
            current_tile=x,
            anchor_data=anchor_data,
            step_idx=step_idx,
            total_steps=total_steps
        )
        
        # Apply the fused result back to the latent state
        x[:, :overlap_size, :] = fused_overlap
        return x

    def _denoise_step(self, x, t_idx, total_inference_steps):
        """
        A single denoising step using DDIM-like logic on the Latent Field.
        Uses the adapted Non-Causal backbone for ε-prediction.
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
            # 1. Predict Noise (ε) using the adapted Non-Causal backbone
            t_tensor = torch.tensor([t_now], device=device).float()
            
            # The model is now treated as a pure noise predictor
            # adapter.predict_noise handles the timestep embedding and head projection
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

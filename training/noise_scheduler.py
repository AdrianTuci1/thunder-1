import torch
import numpy as np
from core.config_manager import THUNDER_CONFIG

class ThunderNoiseScheduler:
    """
    Controls the data degradation curve during training.
    Essential for teaching the model how to denoise across 
    different refinement steps.
    """
    
    def __init__(self, num_train_timesteps=None):
        self.num_train_timesteps = num_train_timesteps or THUNDER_CONFIG["training"]["num_train_timesteps"]
        # Linear beta schedule
        self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Adds noise to the samples according to the schedule.
        """
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        noisy_samples = (sqrt_alphas_cumprod * original_samples + 
                        sqrt_one_minus_alphas_cumprod * noise)
        return noisy_samples

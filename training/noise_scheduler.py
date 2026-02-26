import torch
import numpy as np
from core.config_manager import THUNDER_CONFIG

class ThunderNoiseScheduler:
    """
    Controls the data degradation curve during training.
    Essential for teaching the model how to denoise across 
    different refinement steps.
    """
    
    def __init__(self, num_train_timesteps=None, schedule_type=None):
        self.num_train_timesteps = num_train_timesteps or THUNDER_CONFIG["training"]["num_train_timesteps"]
        self.schedule_type = schedule_type or THUNDER_CONFIG["training"].get("noise_schedule_type", "linear")
        
        if self.schedule_type == "linear":
            self.betas = torch.linspace(1e-4, 0.012, self.num_train_timesteps)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        elif self.schedule_type == "cosine":
            self.alphas_cumprod = self._cosine_schedule(self.num_train_timesteps)
        elif self.schedule_type == "sigmoid":
            self.alphas_cumprod = self._sigmoid_schedule(self.num_train_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule type: {self.schedule_type}")

    def _cosine_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]

    def _sigmoid_schedule(self, timesteps, start=-5, end=5, tau=1.1):
        """
        Optimized Sigmoid: Stays longer in the 'high-information' zone.
        Text diffusion needs more time at low noise levels to learn syntax.
        """
        x = torch.linspace(start, end, timesteps)
        # Aplicăm tau pentru a controla cât de abruptă e tranziția
        alphas_cumprod = torch.sigmoid((x / tau)) 
        # Normalizăm să plece de la ~1.0 la ~0.0
        alphas_cumprod = (alphas_cumprod - alphas_cumprod.min()) / (alphas_cumprod.max() - alphas_cumprod.min())
        return 1.0 - alphas_cumprod

    def add_noise(self, original_samples, noise, timesteps):
        """
        Adds noise with dynamic broadcasting for hidden states.
        """
        device = original_samples.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        t = torch.as_tensor(timesteps, device=device).long()
        
        # Extragem alpha pentru fiecare mostră din batch
        a_cum = self.alphas_cumprod[t]
        
        # Reshape dinamic: [batch, 1, 1] pentru a multiplica [batch, seq, dim]
        while len(a_cum.shape) < len(original_samples.shape):
            a_cum = a_cum.unsqueeze(-1)
            
        sqrt_alphas_cumprod = torch.sqrt(a_cum)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - a_cum)
        
        return (sqrt_alphas_cumprod * original_samples + 
                sqrt_one_minus_alphas_cumprod * noise)
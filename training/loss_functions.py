import torch
import torch.nn.functional as F
from core.config_manager import THUNDER_CONFIG

class HybridLoss:
    """
    Standard Denoising Loss with SNR weighting across the full sequence.
    """
    
    def __init__(self):
        pass

    def calculate_loss(self, predicted_noise, target_noise, timesteps=None):
        """
        Computes the total training loss.
        predicted_noise: [B, L, D]
        target_noise: [B, L, D]
        timesteps: [B] tensor of diffusion steps (normalized or raw)
        """
        # 1. Standard Denoising Loss with SNR weighting
        # MSE is calculated per element first
        mse_loss_raw = F.mse_loss(predicted_noise, target_noise, reduction='none')
        
        if timesteps is not None:
            # SNR weighting: encourages precision in later steps (low noise)
            snr_weights = self._calculate_snr_weights(timesteps)
            # Reshape for broadcasting [B, 1, 1]
            snr_weights = snr_weights.view(-1, 1, 1).to(predicted_noise.device)
            denoising_loss = (mse_loss_raw * snr_weights).mean()
        else:
            denoising_loss = mse_loss_raw.mean()
            
        return denoising_loss

    def _calculate_snr_weights(self, t, max_snr=5.0):
        """
        Time-Weighted Denoising (SNR weighting).
        t: [B] tensor of diffusion timesteps normalized to [0, 1].
        High t = High Noise = Low SNR.
        Low t = Low Noise = High SNR.
        """
        # Heuristic SNR weight: Higher weight for smaller t (cleaner data)
        # This forces the model to be precise about grammar and details.
        # Formula: snr = (1 - t) / (t + 1e-4) -> clipped to [1, max_snr]
        snr = (1.0 - t) / (t + 1e-4)
        return torch.clamp(snr, min=1.0, max=max_snr)

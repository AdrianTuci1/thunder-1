import torch
import torch.nn.functional as F
from core.config_manager import THUNDER_CONFIG

class HybridLoss:
    """
    Standard Denoising Loss (MSE) with SNR weighting across the full sequence,
    combined with a Semantic Anchor Loss to preserve pre-trained intelligence.
    """
    
    def __init__(self, anchor_weight=0.1):
        self.anchor_weight = anchor_weight

    def calculate_loss(self, predicted_noise, target_noise, predicted_x0=None, anchor_x0=None, timesteps=None):
        """
        Computes the total training loss (MSE + Semantic Anchor).
        predicted_noise: [B, L, D] (Predicted epsilon)
        target_noise: [B, L, D] (True epsilon)
        predicted_x0: [B, L, D] (Optional, predicted clean embeddings for anchor loss)
        anchor_x0: [B, L, D] (Optional, original clean embeddings from Llama)
        timesteps: [B] tensor of diffusion steps (normalized or raw)
        """
        # 1. Standard Denoising Loss (MSE on noise)
        mse_loss_raw = F.mse_loss(predicted_noise, target_noise, reduction='none')
        
        if timesteps is not None:
            # SNR weighting: encourages precision in later steps (low noise)
            snr_weights = self._calculate_snr_weights(timesteps)
            snr_weights = snr_weights.view(-1, 1, 1).to(predicted_noise.device)
            denoising_loss = (mse_loss_raw * snr_weights).mean()
        else:
            denoising_loss = mse_loss_raw.mean()
            
        # 2. Semantic Anchor Loss
        # Prevents the diffusion bridge from creating a latent space disconnected from Llama's native intelligence.
        anchor_loss = 0.0
        if predicted_x0 is not None and anchor_x0 is not None:
            # We use Cosine Embedding Loss as it's scale-invariant and focuses on direction (semantic meaning)
            # 1.0 means we want the vectors to be aligned
            target = torch.ones(predicted_x0.shape[0] * predicted_x0.shape[1]).to(predicted_x0.device)
            # Flatten to [B*L, D] for cosine matching
            p_x0_flat = predicted_x0.view(-1, predicted_x0.shape[-1])
            a_x0_flat = anchor_x0.view(-1, anchor_x0.shape[-1])
            
            anchor_loss = F.cosine_embedding_loss(p_x0_flat, a_x0_flat, target)
            
        total_loss = denoising_loss + (self.anchor_weight * anchor_loss)
            
        return total_loss, denoising_loss, anchor_loss

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

import torch
import torch.nn.functional as F
from core.config_manager import THUNDER_CONFIG

class HybridLoss:
    """
    Combines standard Denoising Loss with Boundary Coherence Loss.
    Ensures that parallel tiles remain coherent at their edges.
    """
    
    def __init__(self, boundary_weight=None):
        self.boundary_weight = boundary_weight or THUNDER_CONFIG["training"]["boundary_weight"]

    def calculate_loss(self, predicted_noise, target_noise, timesteps=None, tile_boundaries=None):
        """
        Computes the total training loss.
        predicted_noise: [B, L, D]
        target_noise: [B, L, D]
        timesteps: [B] tensor of diffusion steps (normalized or raw)
        tile_boundaries: Optional tuple (tile_A_overlap, tile_B_overlap) for coherence.
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
        
        if tile_boundaries is not None:
            # 2. Boundary Coherence Loss: penalties for discontinuities at edges
            boundary_loss = self._calculate_boundary_discontinuity(tile_boundaries)
            return denoising_loss + (self.boundary_weight * boundary_loss)
            
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

    def _calculate_boundary_discontinuity(self, boundaries):
        """
        Penalizes sharp changes in hidden states between overlapping tiles.
        boundaries: tuple of tensors (tile_a_overlap, tile_b_overlap)
        Each tensor should be of shape [B, overlap_len, D]
        """
        tile_a_overlap, tile_b_overlap = boundaries
        
        # 1. MSE Loss (Value similarity)
        # Penalizes raw numerical differences in the overlap region
        mse_overlap = F.mse_loss(tile_a_overlap, tile_b_overlap)
        
        # 2. Cosine Similarity (Semantic/Directional similarity)
        # Ensures that "ideas" flow in the same direction across tiles.
        # We flatten to [B*L, D] to calculate per-token similarity
        b, l, d = tile_a_overlap.shape
        flat_a = tile_a_overlap.reshape(-1, d)
        flat_b = tile_b_overlap.reshape(-1, d)
        
        # F.cosine_similarity returns [B*L]
        cos_sim = F.cosine_similarity(flat_a, flat_b, dim=-1)
        # Loss = 1 - average_similarity
        cosine_loss = (1.0 - cos_sim.mean())
        
        # Total boundary loss = MSE + Cosine
        return mse_overlap + cosine_loss

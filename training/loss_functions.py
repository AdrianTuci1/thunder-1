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

    def calculate_loss(self, predicted_noise, target_noise, tile_boundaries=None):
        """
        Computes the total training loss.
        """
        # Standard MSE for denoising
        denoising_loss = F.mse_loss(predicted_noise, target_noise)
        
        if tile_boundaries is not None:
            # Boundary Coherence Loss: penalties for discontinuities at edges
            boundary_loss = self._calculate_boundary_discontinuity(tile_boundaries)
            return denoising_loss + (self.boundary_weight * boundary_loss)
            
        return denoising_loss

    def _calculate_boundary_discontinuity(self, boundaries):
        """
        Penalizes sharp changes in hidden states between overlapping tiles.
        boundaries: tuple of tensors (tile_A_end, tile_B_start)
        """
        tile_a_end, tile_b_start = boundaries
        
        # Calculate consistency loss in the overlapping region
        # High loss if tile A and tile B disagree on the shared context
        overlap_loss = F.mse_loss(tile_a_end, tile_b_start)
        
        return overlap_loss

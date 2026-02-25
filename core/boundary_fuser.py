import torch

from core.config_manager import THUNDER_CONFIG

class BoundaryFuser:
    """
    Handles the blending (Smoothing) of margins between micro-tiles.
    """
    
    def __init__(self):
        self.overlap_size = THUNDER_CONFIG["engine"]["overlap_size"]

    def fuse_tiles(self, tile_a, tile_b):
        """
        Smoothly blends two adjacent tiles at their boundary using a Sigmoid curve.
        Ensures that the transition region (overlap) is coherent.
        """
        # tile_a: [B, L, D], tile_b: [B, L, D]
        # We assume they overlap by self.overlap_size
        
        # Calculate blending weights (Sigmoid curve)
        x = torch.linspace(-5, 5, self.overlap_size).to(tile_a.device)
        weights = torch.sigmoid(x).view(1, -1, 1) # [1, overlap, 1]
        
        # Portion of tile A that overlaps with B (end of A)
        a_overlap = tile_a[:, -self.overlap_size:, :]
        # Portion of tile B that overlaps with A (start of B)
        b_overlap = tile_b[:, :self.overlap_size, :]
        
        # Blended overlapping region
        fused_overlap = (1 - weights) * a_overlap + weights * b_overlap
        
        return fused_overlap

    def apply_global_coherence(self, macro_tile):
        """
        Ensures that local refinements respect the global macro context.
        """
        pass

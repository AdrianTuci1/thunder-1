import torch

from core.config_manager import THUNDER_CONFIG

class BoundaryFuser:
    """
    Handles the blending (Smoothing) of margins between micro-tiles.
    """
    
    def __init__(self):
        self.overlap_size = THUNDER_CONFIG["engine"]["overlap_size"]
        self.coherence_scale = THUNDER_CONFIG["logic"]["scaling"]["coherence_scale"]
        self.anchor_strength = THUNDER_CONFIG["logic"]["scaling"]["fusion_anchor_strength"]

    def get_fusion_mask(self, device):
        """
        Generates a smooth transition mask for anchored denoising.
        """
        x = torch.linspace(-5, 5, self.overlap_size).to(device)
        return torch.sigmoid(x).view(1, -1, 1) # [1, overlap, 1]

    def apply_global_coherence(self, micro_latent, macro_latent_segment):
        """
        Nudges the micro-tile latent towards the macro-tile representation.
        Ensures the local generation doesn't drift away from the global topic.
        """
        if macro_latent_segment is None:
            return micro_latent
            
        # Target: Align micro-latent direction with macro-latent
        # We use a simple nudge (interpolation) towards the macro context
        nudged_latent = (1 - self.coherence_scale) * micro_latent + self.coherence_scale * macro_latent_segment
        return nudged_latent

    def fuse_tiles(self, tile_a, tile_b):
        """
        Legacy fusion: blends two finished tiles. 
        Note: The preferred method is now anchored denoising inside the engine.
        """
        weights = self.get_fusion_mask(tile_a.device)
        
        a_overlap = tile_a[:, -self.overlap_size:, :]
        b_overlap = tile_b[:, :self.overlap_size, :]
        
        fused_overlap = (1 - weights) * a_overlap + weights * b_overlap
        return fused_overlap

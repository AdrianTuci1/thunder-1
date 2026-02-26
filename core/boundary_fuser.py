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
        Uses a directional anchoring approach to maintain semantic flow.
        """
        if macro_latent_segment is None:
            return micro_latent
            
        # Target: Align micro-latent direction with macro-latent
        # 1. Calculate the 'direction' of the macro context
        # 2. Gently steer the micro-latent to align its feature variance with the macro context
        
        # Simple nudge for now, but scaled by coherence_scale
        # In a more advanced version, this would use a residual connection or a small MLP
        nudged_latent = (1 - self.coherence_scale) * micro_latent + self.coherence_scale * macro_latent_segment
        
        # Optional: Normalize to maintain energy levels
        # nudged_latent = nudged_latent * (micro_latent.norm() / nudged_latent.norm())
        
        return nudged_latent

    def fuse_anchored_overlap(self, current_tile, anchor_data, step_idx, total_steps):
        """
        Refined fusion during denoising. 
        Instead of a simple sigmoid, we can dynamically adjust the anchor strength 
        based on the diffusion step.
        """
        overlap_size = self.overlap_size
        device = current_tile.device
        
        # Get smooth mask [1, overlap, 1]
        weights = self.get_fusion_mask(device)
        
        # Scaling anchor strength based on time: 
        # Early steps (high step_idx) -> high noise -> more anchoring
        # Late steps (low step_idx) -> low noise -> more freedom for local details
        time_factor = (total_steps - step_idx) / total_steps
        current_anchor_strength = self.anchor_strength * time_factor
        
        target_overlap = anchor_data[:, -overlap_size:, :]
        current_overlap = current_tile[:, :overlap_size, :]
        
        # Combined weight: fusion mask + temporal anchor strength
        effective_weights = weights * (1 - current_anchor_strength) + current_anchor_strength
        effective_weights = torch.clamp(effective_weights, 0, 1)
        
        fused_overlap = (1 - effective_weights) * target_overlap + effective_weights * current_overlap
        return fused_overlap

from core.config_manager import THUNDER_CONFIG
from core.boundary_fuser import BoundaryFuser

class ThunderDiffusionEngine:
    """
    The core "All-at-Once" crystallization engine.
    Instead of autoregressive generation, it treats tiles as noise fields
    and iteratively refines them into clear text.
    """
    
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.fuser = BoundaryFuser()

    def crystallize_tile(self, tile_data, steps=None, anchor_data=None, macro_context=None):
        """
        Refines a single tile from noise to data with anchored coherence.
        anchor_data: Optional data from the previous tile for boundary fusion.
        macro_context: Optional latent representation of the parent tile.
        """
        if steps is None:
            steps = self.scheduler.get_adaptive_steps(tile_data)
            
        print(f"⚡ Thunder: Crystallizing tile (Steps: {steps}, Anchored: {anchor_data is not None})...")
        
        # Initial cold-start noise field
        current_state = torch.randn_like(tile_data)
        
        for step in range(steps):
            # 1. Prediction step
            current_state = self._denoise_step(current_state, step, steps)
            
            # 2. Apply Global Coherence (Latent Nudge)
            current_state = self.fuser.apply_global_coherence(current_state, macro_context)
            
            # 3. Apply Anchored Fusion (Boundary Constraint)
            if anchor_data is not None:
                current_state = self._apply_anchor(current_state, anchor_data)
            
        return current_state

    def _apply_anchor(self, x, anchor_data):
        """
        Forces the beginning of the current tile to match the end of the 
        previous tile (anchor) within the overlap region.
        """
        overlap_size = self.fuser.overlap_size
        weights = self.fuser.get_fusion_mask(x.device)
        
        # Constrain the beginning of x using the anchor
        # x: [B, L, D], anchor_data: [B, overlap, D]
        target_overlap = anchor_data[:, -overlap_size:, :]
        current_overlap = x[:, :overlap_size, :]
        
        # Smoothly transition from anchor to generated content
        # As we move further into the tile (higher weight), the model has more freedom
        clamped_overlap = (1 - weights) * target_overlap + weights * current_overlap
        
        x[:, :overlap_size, :] = clamped_overlap
        return x

    def _denoise_step(self, x, current_step, total_steps):
        """
        A single denoising step using the Phi-4 backbone.
        """
        # Calculate scaling factors based on the schedule
        t = torch.tensor([current_step / total_steps])
        
        with torch.no_grad():
            # Prediction from the backbone model
            model_output = self.model(x).logits  # Simplified
            
            # Update latent: x_{t-1} = refinement(x_t, model_output)
            # Use boundary_weight from config for refinement scale
            refinement_scale = THUNDER_CONFIG["training"]["boundary_weight"]
            refinement_step = model_output * refinement_scale
            new_state = x - refinement_step
            
            return new_state

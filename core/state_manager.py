import torch

class GlobalStateManager:
    """
    Manages the "Global Latent Field" for the Thunder diffusion engine.
    Handles the persistence and synchronization of noisy/denoised latent 
    vectors across 16k+ token boundaries.
    """
    
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.latent_field_cache = {}

    def capture_latent_state(self, tile_id, latent_tensor):
        """
        Caches the partially denoised latent state for a specific tile.
        """
        self.latent_field_cache[tile_id] = latent_tensor.detach().clone()

    def get_latent_state(self, tile_id):
        """
        Retrieves the cached latent state for a specific tile.
        """
        return self.latent_field_cache.get(tile_id)

    def synchronize_diffusion_field(self, meso_id, micro_tiles):
        """
        Ensures semantic continuity in the latent field across parallel tiles.
        Propagates the boundary 'anchor' latents from tile i-1 to tile i.
        """
        print(f"⚡ Thunder: Synchronizing Diffusion Field for Meso-tile {meso_id}...")
        
        for i in range(1, len(micro_tiles)):
            prev_id = micro_tiles[i-1]["id"]
            curr_id = micro_tiles[i]["id"]
            
            prev_latent = self.get_latent_state(prev_id)
            if prev_latent is not None:
                # The end-boundary of tile (i-1) becomes the starting anchor for tile i
                # This ensures the 'noise-to-token' transition is smooth.
                self.latent_field_cache[curr_id + "_anchor"] = prev_latent.detach().clone()

    def get_latent_anchor(self, tile_id):
        """
        Retrieves the latent anchor for the current tile's diffusion process.
        """
        return self.latent_field_cache.get(tile_id + "_anchor")

    def clear_field(self):
        self.latent_field_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

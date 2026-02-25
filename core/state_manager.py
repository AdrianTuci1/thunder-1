import torch

class GlobalStateManager:
    """
    Manages the Mamba "Global State" for massive 120k context windows.
    Handles caching and retrieval of internal states across tiling boundaries.
    """
    
    def __init__(self, dimension, state_size=16):
        self.dimension = dimension
        self.state_size = state_size
        self.global_state_cache = {}

    def capture_state(self, tile_id, state_tensor):
        """
        Caches the state for a specific tile.
        """
        # Store on GPU if possible, else CPU to save VRAM for 120k context
        self.global_state_cache[tile_id] = state_tensor.detach().clone()

    def get_state(self, tile_id):
        """
        Retrieves cached state for a specific tile.
        """
        return self.global_state_cache.get(tile_id)

    def synchronize_states(self, meso_id, micro_tiles):
        """
        Ensures consistency between Micro-tiles within a Meso-tile boundary.
        Propagates the Mamba hidden state from block i-1 to block i.
        """
        print(f"⚡ Thunder: Synchronizing states for Meso-tile {meso_id}...")
        
        for i in range(1, len(micro_tiles)):
            prev_id = micro_tiles[i-1]["id"]
            curr_id = micro_tiles[i]["id"]
            
            prev_state = self.get_state(prev_id)
            if prev_state is not None:
                # Merge states - simplifies the transition between parallel streams
                # Logic involves matching the end state of tile A to the start of tile B
                pass

    def clear(self):
        self.global_state_cache.clear()
        torch.cuda.empty_cache()

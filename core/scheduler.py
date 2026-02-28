from core.config_manager import THUNDER_CONFIG

class ThunderScheduler:
    """
    ADAPTIVE SCHEDULER: Decides the number of denoising steps based on 
    tile complexity and hardware constraints.
    """
    
    def __init__(self, default_steps=None, min_steps=None, max_steps=None):
        self.default_steps = default_steps or THUNDER_CONFIG["logic"]["default_steps"]
        self.min_steps = min_steps or THUNDER_CONFIG["logic"]["min_steps"]
        self.max_steps = max_steps or THUNDER_CONFIG["logic"]["max_steps"]
        self.mode_configs = THUNDER_CONFIG["logic"]["modes"]
        self.scaling_configs = THUNDER_CONFIG["logic"]["scaling"]

    def calculate_steps(self, tile_node=None, mode=None, predicted_length=100):
        """
        DYNAMIC STEPS: Determines iterations based on mode, complexity, and length.
        """
        import math
        import random
        
        # 1. Base steps from mode or default
        if mode and mode in self.mode_configs:
            base_steps = self.mode_configs[mode]["base"]
            limit_steps = self.mode_configs[mode]["max"]
        else:
            base_steps = self.default_steps
            limit_steps = self.max_steps

        # 2. Depth factor: Leaf nodes are high-fidelity
        if hasattr(tile_node, 'is_leaf') and tile_node.is_leaf:
            base_steps = int(base_steps * 1.2)
            
        # 3. Complexity factor (Placeholder for entropy-based scoring)
        complexity_score = random.uniform(0.8, 1.2)
        
        # 4. Length scaling: log(predicted_length) factor
        # Using log10(len) + 0.5 as a multiplier (e.g. 100 tokens -> 2+0.5 = 2.5x base? too much, let's use a weight)
        length_factor = 1.0 + (math.log10(max(1, predicted_length)) * self.scaling_configs["length_weight"])
        
        dynamic_steps = int(base_steps * complexity_score * length_factor)
        
        # Clamp between hardware min and mode-specific max
        return max(self.min_steps, min(limit_steps, dynamic_steps))

    def get_adaptive_steps(self, tile_data):
        """
        Heuristic to determine optimal steps for a tile.
        Complexity can be based on variance or entropy of the latent representation.
        """
        # Placeholder for complexity analysis logic
        complexity = self._calculate_complexity(tile_data)
        
        if complexity > 0.8:
            return self.max_steps
        elif complexity < 0.2:
            return self.min_steps
        return self.default_steps

    def _calculate_complexity(self, data):
        """
        Measures the information density of the tile.
        """
        # Mock complexity factor
        return 0.5

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

    def calculate_steps(self, mode="fast", anchor_len=0):
        """
        DYNAMIC STEPS (Mercury 1): Determines optimal iterations based on mode and prompt complexity.
        """
        import math
        
        # 1. Base steps from mode or default
        if mode and mode in self.mode_configs:
            base_steps = self.mode_configs[mode]["base"]
            limit_steps = self.mode_configs[mode]["max"]
        else:
            base_steps = self.default_steps
            limit_steps = self.max_steps

        # 2. Length scaling: log(anchor_len) factor
        # Longer prompts usually imply more complex logic/reasoning required.
        length_factor = 1.0 + (math.log10(max(1, anchor_len)) * self.scaling_configs["length_weight"])
        
        dynamic_steps = int(base_steps * length_factor)
        
        # Clamp between hardware min and mode-specific max
        return max(self.min_steps, min(limit_steps, dynamic_steps))

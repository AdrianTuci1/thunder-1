from core.config_manager import THUNDER_CONFIG

class LoRAConfig:
    """
    Defines optimized Rank & Alpha parameters for parallel diffusion.
    """
    
    def __init__(self, rank=None, alpha=None):
        self.rank = rank or THUNDER_CONFIG["training"]["lora_rank"]
        self.alpha = alpha or THUNDER_CONFIG["training"]["lora_alpha"]
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    def get_config(self):
        """
        Returns parameters for FastLanguageModel.get_peft_model.
        """
        return {
            "r": self.rank,
            "target_modules": self.target_modules,
            "lora_alpha": self.alpha,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
        }

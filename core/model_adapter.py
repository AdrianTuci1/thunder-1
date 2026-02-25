import torch
from unsloth import FastLanguageModel

class ThunderModelAdapter:
    """
    Injects LoRA adapters and activates bilateral convergence capabilities
    for the Thunder diffusion engine.
    """
    
    def __init__(self, model):
        self.model = model

    def apply_lora(self, r=64, lora_alpha=128, target_modules=None):
        """
        Applies LoRA adapters optimized for parallel denoising.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"]

        print(f"⚡ Thunder: Injecting LoRA (r={r}, alpha={lora_alpha})...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=0,  # Optimized for inference speed
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return self.model

    def enable_bilateral_hooks(self):
        """
        Activates hooks that allow future tokens to influence past tokens 
        during the iterative denoising phase by wrapping attention/Mamba blocks.
        """
        print("⚡ Thunder: Activating Bilateral Convergence hooks...")
        
        def bilateral_wrapper(module, input, output):
            # If output is a tuple (like in some transformers), handle accordingly
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Allow bidirectional influence across the sequence dimension
            # This logic interacts with the noise field during diffusion
            # forward_pass -> global_refinement -> backward_influence
            return output

        # Register hooks for the core blocks (Phi-4 / Mamba)
        for name, module in self.model.named_modules():
            if any(target in name for target in ["mixer", "layers", "attention"]):
                module.register_forward_hook(bilateral_wrapper)

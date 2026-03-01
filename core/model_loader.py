from unsloth import FastLanguageModel
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.config_manager import THUNDER_CONFIG

class ThunderModelLoader:
    """
    Handles loading of Phi-4/Mamba models using Unsloth for optimized 4-bit/BF16 inference.
    """
    
    def __init__(self, model_name=None):
        self.model_name = model_name # Store the model_name passed to __init__
        self.max_seq_length = THUNDER_CONFIG["engine"]["max_seq_len"]
        self.model = None
        self.tokenizer = None

    def load_model(self, load_in_4bit=True, inference_mode=True):
        """
        Loads the model and tokenizer.
        """
        import torch
        # Priority: constructor arg > config > default
        model_to_load = self.model_name or THUNDER_CONFIG["engine"].get("model_path")
        
        print(f"⚡ Thunder: Loading model {model_to_load}...")
        
        # Check if it's a LLaDA model to decide loading strategy
        is_llada = "LLaDA" in model_to_load
        
        if is_llada:
            # LLaDA might need standard transformers if unsloth hasn't patched it yet
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                load_in_4bit=load_in_4bit,
                device_map="auto"
            )
        else:
            # Fallback for Llama/other models
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_to_load,
                max_seq_length=self.max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )
        
        # 3. Adapt model for PrefixLM Diffusion
        from core.diffusion_model import PrefixLMDiffusionAdapter
        adapter = PrefixLMDiffusionAdapter(self.model)
        self.model = adapter.adapt_for_diffusion(checkpoint_path=model_to_load)
        
        if inference_mode:
            # Enable faster inference 
            FastLanguageModel.for_inference(self.model)
        
        return self.model, self.tokenizer

    def get_model_info(self):
        if self.model:
            return {
                "params": self.model.num_parameters(),
                "config": self.model.config.to_dict(),
                "device": next(self.model.parameters()).device
            }
        return None

import torch
from unsloth import FastLanguageModel
import os

from core.config_manager import THUNDER_CONFIG

class ThunderModelLoader:
    """
    Handles loading of Phi-4/Mamba models using Unsloth for optimized 4-bit/BF16 inference.
    """
    
    def __init__(self, model_name=None):
        self.model_name = model_name or "unsloth/Llama-3.2-3B-Instruct"
        self.max_seq_length = THUNDER_CONFIG["engine"]["max_seq_len"]
        self.model = None
        self.tokenizer = None

    def load_model(self, load_in_4bit=True):
        """
        Loads the model and tokenizer with 4-bit quantization and BF16 support.
        """
        # Prioritize model_path from config if specified
        model_to_load = THUNDER_CONFIG["engine"].get("model_path") or self.model_name
        
        print(f"⚡ Thunder: Loading model {model_to_load} with {self.max_seq_length} context...")
        
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
        self.model = adapter.adapt_for_diffusion()
        
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

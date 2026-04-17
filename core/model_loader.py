import torch

from core.config_manager import THUNDER_CONFIG


class ThunderModelLoader:
    """
    Loads either the new from-scratch bidirectional diffusion LM or the legacy
    pretrained path, depending on configuration.
    """

    def __init__(self, model_name=None):
        self.model_name = model_name
        self.max_seq_length = THUNDER_CONFIG["engine"]["max_seq_len"]
        self.model = None
        self.tokenizer = None

    def _default_device(self):
        if THUNDER_CONFIG["engine"].get("device") == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(THUNDER_CONFIG["engine"]["device"])

    def _default_dtype(self):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def _load_scratch_model(self):
        from transformers import AutoTokenizer

        from core.scratch_dllm import ScratchDLMConfig, ThunderScratchDiffusionLM

        tokenizer_name = self.model_name or THUNDER_CONFIG["engine"]["tokenizer_name"]
        print(f"⚡ Thunder: Loading tokenizer {tokenizer_name} for from-scratch dLLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Suppress warnings about sequence length during block packing
        self.tokenizer.model_max_length = 100000

        import inspect
        sig = inspect.signature(ScratchDLMConfig)
        valid_keys = sig.parameters.keys()
        
        full_model_config = {
            **THUNDER_CONFIG["model"],
            "vocab_size": len(self.tokenizer),
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_seq_len": THUNDER_CONFIG["engine"]["max_seq_len"],
        }
        
        # Filtram doar cheile valide pentru dataclass
        model_config = {k: v for k, v in full_model_config.items() if k in valid_keys}
        
        config = ScratchDLMConfig(**model_config)
        self.model = ThunderScratchDiffusionLM(config, diffusion_steps=THUNDER_CONFIG["diffusion"]["steps"])
        self.model.to(device=self._default_device(), dtype=self._default_dtype())

        approx_params = config.estimate_parameter_count()
        print(
            "⚡ Thunder: Scratch dLLM initialized "
            f"(estimated params ~{approx_params / 1_000_000:.1f}M, actual params {self.model.num_parameters() / 1_000_000:.1f}M)."
        )
        return self.model, self.tokenizer

    def _load_pretrained_model(self, load_in_4bit=True):
        from unsloth import FastLanguageModel

        print("⚠️  Thunder: Falling back to legacy pretrained adapter path.")
        model_to_load = THUNDER_CONFIG["engine"].get("model_path") or self.model_name or "Qwen/Qwen3.5-9B"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_to_load,
            max_seq_length=self.max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            use_gradient_checkpointing="unsloth" if THUNDER_CONFIG["hardware"].get("gradient_checkpointing") else False,
        )

        from core.diffusion_model import PrefixLMDiffusionAdapter

        adapter = PrefixLMDiffusionAdapter(self.model)
        self.model = adapter.adapt_for_diffusion()
        FastLanguageModel.for_inference(self.model)
        return self.model, self.tokenizer

    def load_model(self, load_in_4bit=True):
        model_source = THUNDER_CONFIG["engine"].get("model_source", "scratch")
        if model_source == "scratch":
            return self._load_scratch_model()
        return self._load_pretrained_model(load_in_4bit=load_in_4bit)

    def get_model_info(self):
        if self.model is None:
            return None

        config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else {}
        return {
            "params": self.model.num_parameters() if hasattr(self.model, "num_parameters") else None,
            "config": config,
            "device": str(next(self.model.parameters()).device),
        }

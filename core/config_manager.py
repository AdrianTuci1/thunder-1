"""
THUNDER MASTER CONFIGURATION
Centralized dictionary for all performance tuning and architectural parameters.
"""

THUNDER_CONFIG = {
    # ----------------------------------------------------------------------
    # 1. CORE ENGINE & HARDWARE
    # ----------------------------------------------------------------------
    "engine": {
        "model_path": "Qwen/Qwen3.5-9B",
        "max_seq_len": 32768,      # Input context (Prompt)
        "max_gen_len": 8192,       # Crystallization limit (Output)
    },
    
    "hardware": {
        # Profile: RTX 4090 (24GB VRAM)
        "load_in_4bit": True,
        "batch_size": 1,
        "grad_accum": 16,
        
        # Profile: NVIDIA A100 (80GB VRAM) - Uncomment to activate
        # "load_in_4bit": False,   # Native BF16 for maximum precision
        # "batch_size": 8,
        # "grad_accum": 2,

        "bf16_support": True,
        "fused_kernels": True,
        "flash_attention": True,
        "gradient_checkpointing": True,
        "stream_count": 64,      # Can be increased to 128 on A100
    },
    
    # ----------------------------------------------------------------------
    # 2. DIFFUSION CORE (RESONANCE FIELD)
    # ----------------------------------------------------------------------
    "diffusion": {
        "steps": 2000,                # Training timesteps (T)
        "schedule": "sigmoid",        # Transitioned to Sigmoid for smoother text logic
        "cfg_drop_rate": 0.1,         # Slightly lower dropout for 9B stability
    },
    
    # ----------------------------------------------------------------------
    # 3. TRAINING & ADAPTATION (LoRA)
    # ----------------------------------------------------------------------
    "training": {
        "lora_rank": 128,              
        "lora_alpha": 512,            # Increased for stronger 9B adaptation
        
        "learning_rate": 4e-5,        # Lowered for 9B model stability
        "max_steps": 20000,           # Target for solid AR-to-Diffusion transfer
        "warmup_steps": 200,      
        "optim": "paged_adamw_8bit",  # More stable for long context
        "lr_scheduler": "cosine",
        "output_dir": "./thunder_qwen_32k",
        "save_steps": 500,       
        "save_total_limit": 5,   
        "logging_steps": 1,      
        "seed": 3407,            
    },
    
    # ----------------------------------------------------------------------
    # 4. DATA PIPELINE
    # ----------------------------------------------------------------------
    "pipeline": {
        "dataset_name": [
            "Open-Thoughts/Open-Thoughts-0.5B", # Dense CoT reasoning for Diffusion-LM
            "Open-Orca/SlimOrca",
            "nickrosh/Evol-Instruct-Code-80k-v1",
            "qwedsacf/competition_math",
            "zai-org/LongAlign-10k"
        ],
        "dataset_ratios": [0.25, 0.25, 0.15, 0.15, 0.20], # Rebalanced for reasoning density
        "num_proc": 4,           
        "packing": True,         # Enables Constant Length Packing
    },
    
    # ----------------------------------------------------------------------
    # 5. INFERENCE LOGIC (MERCURY MODES)
    # ----------------------------------------------------------------------
    "logic": {
        "modes": {
            "instant": {"base": 8,   "max": 15}, # Optimized for 8-step target
            "fast":    {"base": 12,  "max": 30},
            "thinking":{"base": 30,  "max": 100}
        },
        "scaling": {
            "length_weight": 0.5, # Multiplier for log10(length)
        },
        "default_steps": 25,
        "min_steps": 5,
        "max_steps": 100,
    },
    
    # ----------------------------------------------------------------------
    # 6. SERVER & INFERENCE
    # ----------------------------------------------------------------------
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_token": "thunder-secret-at-2026",
    }
}


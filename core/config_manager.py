"""
THUNDER MASTER CONFIGURATION
Centralized dictionary for all performance tuning and architectural parameters.
"""

THUNDER_CONFIG = {
    # ----------------------------------------------------------------------
    # 1. CORE ENGINE & HARDWARE
    # ----------------------------------------------------------------------
    "engine": {
        "model_path": "unsloth/Llama-3.2-3B-Instruct",
        "max_seq_len": 2048,
    },
    
    "hardware": {
        "load_in_4bit": True,    # Enables 4-bit quantization via Unsloth to fit 24GB VRAM
        "bf16_support": True,    # Uses BFloat16 if supported by hardware (sm_8x / sm_9x)
        "fused_kernels": True,   # Enables kernel fusion in kernels/paged_clamping.cu
        "stream_count": 64,      # Number of parallel CUDA Streams
    },
    
    # ----------------------------------------------------------------------
    # 2. DIFFUSION ARCHITECTURE (MERCURY 1)
    # ----------------------------------------------------------------------
    "diffusion": {
        "diffusion_steps": 2000,       # T (Diffusion Process steps)
        "noise_schedule_type": "sqrt", # Custom text diffusion schedule
        "cfg_drop_rate": 0.15,         # Prompt dropout for Classifier-Free Guidance
    },
    
    # ----------------------------------------------------------------------
    # 3. TRAINING & FINE-TUNING
    # ----------------------------------------------------------------------
    "training": {
        # LoRA Settings
        "lora_rank": 128,        # Rank (r) of LoRA adapters
        "lora_alpha": 256,       # Scaling factor (alpha)
        
        # Optimization
        "learning_rate": 3.0e-5, # Lowered for stability in Diffusion Fine-tuning   
        "batch_size": 4,         # Reduced to 4 to fix OOM on 24GB VRAM
        "grad_accum": 8,         # Increased to 8 to keep Effective Batch Size at 32
        "max_steps": 55000,       
        "warmup_steps": 1000,      
        "optim": "adamw_8bit",   
        "weight_decay": 0.01,    
        "lr_scheduler": "cosine",
        "max_grad_norm": 1.0,    
        
        # Output & State
        "output_dir": "./thunder_prefixlm_llama",
        "save_steps": 200,       
        "save_total_limit": 3,   
        "logging_steps": 1,      
        "seed": 3407,            
    },
    
    # ----------------------------------------------------------------------
    # 4. DATA PIPELINE
    # ----------------------------------------------------------------------
    "pipeline": {
        "dataset_name": [
            "Open-Orca/SlimOrca",
            "nickrosh/Evol-Instruct-Code-80k-v1",
            "qwedsacf/competition_math",
            "nomic-ai/gpt4all-j-prompt-generations",
            "zai-org/LongAlign-10k",
            "nohurry/Opus-4.6-Reasoning-3000x-filtered"
        ],
        "dataset_ratios": [0.15, 0.15, 0.10, 0.20, 0.20, 0.20], # Rebalanced for text logic
        "num_proc": 4,           
        "packing": True,         # Enables Constant Length Packing
    },
    
    # ----------------------------------------------------------------------
    # 5. INFERENCE LOGIC (MERCURY MODES)
    # ----------------------------------------------------------------------
    "logic": {
        "modes": {
            "instant": {"base": 10,  "max": 25},
            "fast":    {"base": 20,  "max": 50},
            "thinking":{"base": 50,  "max": 100}
        },
        "scaling": {
            "length_weight": 0.5, # Multiplier for log10(length)
        },
        "default_steps": 25,
        "min_steps": 5,
        "max_steps": 100,
        "internal_threshold": 0.5,
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


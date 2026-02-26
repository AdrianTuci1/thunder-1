"""
THUNDER MASTER CONFIGURATION
Centralized dictionary for all performance tuning and architectural parameters.
"""

THUNDER_CONFIG = {
    # --- ENGINE ---
    "engine": {
        "max_seq_len": 16384,    # TARGET: 16k testing window natively
    },
    
    # --- HARDWARE & ACCELERATION (RTX 4090 / Ada Lovelace) ---
    "hardware": {
        "stream_count": 64,      # Number of parallel CUDA Streams (higher = more GPU saturation)
        "load_in_4bit": True,    # Enables 4-bit quantization via Unsloth to fit 24GB VRAM
        "bf16_support": True,    # Uses BFloat16 if supported by hardware (sm_8x / sm_9x)
        "fused_kernels": True,   # Enables kernel fusion in kernels/fused_diffusion.cu
    },
    
    # --- DIFFUSION LOGIC ---
    "logic": {
        "max_steps": 100,        # Maximum crystallization steps for high-fidelity output
        "min_steps": 5,          # Minimum steps for instant-pass generation
        "default_steps": 50,     # Standard steps for balanced performance
        "internal_threshold": 0.8, # Gating threshold for routing query internally
        
        # Mode-specific baseline steps
        "modes": {
            "instant": {"base": 5, "max": 14},
            "fast": {"base": 15, "max": 25},
            "thinking": {"base": 30, "max": 50}
        },
        
        # Scaling factors
        "scaling": {
            "complexity_weight": 0.5,
            "length_weight": 0.3,   # log-scale multiplier for predicted response length
        }
    },
    
    # --- TOKEN SAMPLING ---
    "sampling": {
        "temperature": 0.7,      # Randomness control (higher = more creative)
        "top_p": 0.9,            # Nucleus sampling threshold
        "top_k": 50,             # Top-K sampling filter
    },
    
    # --- TRAINING & LORA ---
    "training": {
        "lora_rank": 64,        # Rank (r) of LoRA adapters (higher = more complex reasoning)
        "lora_alpha": 128,       # Scaling factor (alpha) for LoRA updates
        "learning_rate": 2e-4,   # Initial learning rate for SFT sau 5e-5 daca nu scade loss-ul
        "batch_size": 2,         # Per-device training batch size
        "grad_accum": 4,         # Gradient accumulation steps
        "max_steps": 60,         # Total number of training steps ( use 1200 )
        "warmup_steps": 5,       # Linear warmup steps
        "logging_steps": 1,      # Step interval for logging progress
        "optim": "adamw_8bit",   # Optimizer type (8-bit AdamW for VRAM efficiency)
        "weight_decay": 0.01,    # Weight decay for regularization
        "lr_scheduler": "cosine", # Learning rate decay schedule
        "seed": 3407,            # Random seed for reproducibility
        "output_dir": "./thunder_finetuned", # Directory for saving model weights
        
        # Data Pipeline
        "dataset_name": [
            "HuggingFaceH4/ultrafeedback_binarized",
            "Open-Orca/SlimOrca",
            "THUDM/LongAlign"
        ],
        "dataset_ratios": [0.5, 0.3, 0.2], # 50% alignment, 30% reasoning, 20% long context
        "num_proc": 4,           # Increased for faster mapping
        "packing": True,         # Enables Constant Length Packing for 120k tokens
        
        # Checkpointing
        "save_steps": 200,       # Save model every 200 steps
        "save_total_limit": 3,   # Keep only the last 3 checkpoints to save disk
        
        # Noise & Loss
        "num_train_timesteps": 1000, # Steps in the noise degradation schedule
        "noise_schedule_type": "cosine", # "linear", "cosine", or "sigmoid"
    },
    
    # --- SERVER & INTERFACE ---
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_token": "thunder-secret-at-2026" # Default token for security
    }
}

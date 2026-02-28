"""
THUNDER MASTER CONFIGURATION
Centralized dictionary for all performance tuning and architectural parameters.
"""

THUNDER_CONFIG = {
    # --- ENGINE ---
    "engine": {
        "max_seq_len": 2048,    # TARGET: 16k testing window
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
        "max_steps": 70,        # Maximum crystallization steps for high-fidelity output
        "min_steps": 5,          # Minimum steps for instant-pass generation
        "default_steps": 14,     # Standard steps for balanced performance
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
        "lora_rank": 128,        # Rank (r) for 3% param update on 4B (Architecture Shift)
        "lora_alpha": 256,       # Scaled with r (alpha = 2 * r)
        "learning_rate": 8e-5,   # Pivoted LR for stability after step 600
        "batch_size": 1,         # Per-device training batch size (reduced for OOM)
        "grad_accum": 8,         # Gradient accumulation steps (increased to keep effective BS)
        "max_steps": 1600,       # Total number of training steps
        "warmup_steps": 50,      # Increased warmup for stability
        "logging_steps": 1,      # Step interval for logging progress
        "optim": "adamw_8bit",   # Optimizer type (8-bit AdamW for VRAM efficiency)
        "weight_decay": 0.01,    # Weight decay for regularization
        "lr_scheduler": "cosine", # Learning rate decay schedule
        "seed": 3407,            # Random seed for reproducibility
        "output_dir": "./thunder_finetuned", # Directory for saving model weights
        
        # Data Pipeline (Pivot @ Step 600)
        "dataset_name": [
            "Open-Orca/SlimOrca",
            "nickrosh/Evol-Instruct-Code-80k-v1",
            "open-web-math/open-web-math",
            "nomic-ai/gpt4all-j-prompt-generations",
            "THUDM/LongAlign-10k"
        ],
        "dataset_ratios": [0.1, 0.15, 0.45, 0.2, 0.1], # 45% Math for gradient anchor
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

"""
THUNDER MASTER CONFIGURATION
Centralized dictionary for all performance tuning and architectural parameters.
"""

THUNDER_CONFIG = {
    # --- ENGINE ---
    "engine": {
        "max_seq_len": 2048,
        "model_path": "unsloth/Llama-3.2-3B-Instruct", # Switching to Llama 3.2 3B as requested
    },
    
    # --- HARDWARE & ACCELERATION (RTX 4090 / Ada Lovelace) ---
    "hardware": {
        "stream_count": 64,      # Number of parallel CUDA Streams (higher = more GPU saturation)
        "load_in_4bit": True,    # Enables 4-bit quantization via Unsloth to fit 24GB VRAM
        "bf16_support": True,    # Uses BFloat16 if supported by hardware (sm_8x / sm_9x)
        "fused_kernels": True,   # Enables kernel fusion in kernels/fused_diffusion.cu
    },
    
    # --- TOKEN SAMPLING ---
    "sampling": {
        "temperature": 0.7,      # Randomness control (higher = more creative)
        "top_p": 0.9,            # Nucleus sampling threshold
        "top_k": 50,             # Top-K sampling filter
    },
    
    # --- TRAINING & LORA ---
    "training": {
        "lora_rank": 128,        # Rank (r) of LoRA adapters (reduced from 64 for VRAM)
        "lora_alpha": 256,       # Scaling factor (alpha) for LoRA updates
        "learning_rate": 8e-5,   # Increased for faster convergence from checkpoint
        "batch_size": 2,         # Per-device training batch size (reduced for OOM)
        "grad_accum": 8,         # Gradient accumulation steps (increased to keep effective BS)
        "max_steps": 1500,       # Total steps (next phase)
        "warmup_steps": 50,       # Linear warmup steps
        "logging_steps": 1,      # Step interval for logging progress
        "optim": "adamw_8bit",   # Optimizer type (8-bit AdamW for VRAM efficiency)
        "weight_decay": 0.01,    # Weight decay for regularization
        "lr_scheduler": "cosine", # Learning rate decay schedule
        "seed": 3407,            # Random seed for reproducibility
        "output_dir": "./thunder_prefixlm_llama", # Fresh directory for the new architecture
        "max_grad_norm": 1.0,    # Prevents gradient explosion during diffusion
        
        # Data Pipeline
        "dataset_name": [
            "Open-Orca/SlimOrca",
            "nickrosh/Evol-Instruct-Code-80k-v1",
            "qwedsacf/competition_math",
            "nomic-ai/gpt4all-j-prompt-generations",
            "zai-org/LongAlign-10k"
        ],
        "dataset_ratios": [0.15, 0.15, 0.35, 0.2, 0.15], # Reduced Math, increased Orca/LongAlign
        "num_proc": 4,           # Increased for faster mapping
        "packing": True,         # Enables Constant Length Packing for 120k tokens
        
        # Checkpointing
        "save_steps": 200,       # Save model every 200 steps
        "save_total_limit": 3,   # Keep only the last 3 checkpoints to save disk
        
        # Noise & Loss
        "num_train_timesteps": 2000, # Increased to 2000 for paper alignment
        "noise_schedule_type": "sqrt", # Switched to Diffusion-LM's sqrt schedule
    },
    
    # --- SERVER & INTERFACE ---
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_token": "thunder-secret-at-2026" # Default token for security
    }
}

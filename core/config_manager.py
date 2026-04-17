"""
THUNDER MASTER CONFIGURATION
Focused on a from-scratch diffusion language model target that can train on A100
and serve inference on RTX 4090 with short fast-step decoding.
"""

THUNDER_CONFIG = {
    # ------------------------------------------------------------------
    # 1. ENGINE TARGET
    # ------------------------------------------------------------------
    "engine": {
        "model_source": "scratch",
        "tokenizer_name": "HuggingFaceTB/SmolLM2-135M",
        "max_seq_len": 2048,           # Increased for Big Run
        "max_gen_len": 2048,
        "device": "auto",
    },

    # ------------------------------------------------------------------
    # 2. FROM-SCRATCH MODEL
    # ------------------------------------------------------------------
    "model": {
        "vocab_size": 49152,           # Locked: SmolLM2-135M tokenizer. Do not change after pretraining starts.
        "embedding_dim": 1152,           # Token space used for clamping and logits.
        "latent_dim": 1152,              # Compressed denoising space for cheaper diffusion steps.
        "ffn_hidden_size": 4608,
        "num_layers": 24,
        "num_attention_heads": 16,
        "num_kv_heads": 4,             # GQA: 4:1 ratio
        "dropout": 0.0,
        "max_seq_len": 2048,
        "pad_token_id": 0,
        "self_conditioning": True,
        "use_rope": True,              # Enable Rotary Positional Embeddings
        "rope_theta": 500000.0,         # Adjusted for 8k+ context stability
        "latent_bridge": {
            "enabled": True,
            "compression_ratio": 1.2,
        },
    },

    # ------------------------------------------------------------------
    # 3. HARDWARE PROFILES
    # ------------------------------------------------------------------
    "hardware": {
        "load_in_4bit": False,
        "batch_size": 8,               # Reduced from 16 to avoid OOM, still 2x original
        "grad_accum": 16,               # Increased to keep EBS = 128
        "bf16_support": True,
        "gradient_checkpointing": True,
        "flash_attention": True,
        "fused_kernels": False,
        "stream_count": 32,
        "target_train_gpu": "A100 80GB",
        "target_inference_gpu": "RTX 4090 24GB",
    },

    # ------------------------------------------------------------------
    # 4. DIFFUSION CORE
    # ------------------------------------------------------------------
    "diffusion": {
        "steps": 256,
        "schedule": "sigmoid",
        "cfg_drop_rate": 0.10,
        "teacher_steps": 32,
        "fast_steps": 5,
        "thinking_steps": 16,
    },

    # ------------------------------------------------------------------
    # 5. TRAINING
    # ------------------------------------------------------------------
    "training": {
        "learning_rate": 1.2e-4,       # Reduced for better stability after spikes
        "weight_decay": 0.1,
        "warmup_steps": 3000,
        "curriculum_stage_steps": 1250,
        "epochs": 1,
        "max_steps": 150000, 
        "output_dir": "./runs/thunder_v1_850M_production",
        "save_steps": 5000,
        "save_interval_hours": 2,      # Added per user request
        "logging_steps": 1,
        "preview_steps": 1000,
        "save_total_limit": 5,
        "t_round_penalty": 0.0,
        "resume_from": "./runs/thunder_v1_850M_production/checkpoint-8273",
        "noise_sampling_mode": "biased",
        "noise_sampling_range": [0, 40],
        "lr_schedule_type": "thunder_warmdown",
        "warmdown_constant_steps": 10000,
        "seed": 3407,
        "pipeline_key": "pretrain_hf_datasets",
        "max_train_blocks": 4882813,   
        "max_documents_per_dataset": 1000000,
        "use_wandb": True,             # Generate automatic WandB links
        "wandb_project": "thunder-dllm",
    },

    # ------------------------------------------------------------------
    # 6. DATA PIPELINE
    # ------------------------------------------------------------------
    "pipeline": {
        "num_proc": 4,
        "packing": True,
        "block_size": 2048,
        "curriculum_lengths": [512, 1024, 2048],
        "eos_between_documents": True,
        "shuffle_seed": 3407,
        "pretrain_hf_datasets": [
            {
                "path": "identity_data/thunder_identity.jsonl",
                "split": "train",
                "format": "prompt_response",
                "weight": 0.03,
            },
            {
                "path": "HuggingFaceTB/cosmopedia-v2",
                "name": "cosmopedia-v2",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.27,
            },
            {
                "path": "HuggingFaceTB/smollm-corpus",
                "name": "fineweb-edu-dedup",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.45,
            },
            {
                "path": "open-web-math/open-web-math",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.15,
            },
            {
                "path": "codeparrot/codeparrot-clean",
                "split": "train",
                "format": "text",
                "text_field": "content",
                "weight": 0.10,
            },
        ],
        "sft_hf_datasets": [
            {
                "path": "Open-Orca/SlimOrca",
                "split": "train",
                "format": "conversations",
                "messages_field": "conversations",
                "streaming": True,
                "weight": 0.35,
            },
            {
                "path": "HuggingFaceH4/ultrafeedback_binarized",
                "split": "train_sft",
                "format": "messages",
                "messages_field": "messages",
                "streaming": True,
                "weight": 0.35,
            },
            {
                "path": "open-thoughts/OpenThoughts-114k",
                "split": "train",
                "format": "conversations",
                "messages_field": "conversations",
                "streaming": True,
                "weight": 0.20,
            },
            {
                "path": "codeparrot/codeparrot-clean",
                "split": "train",
                "format": "text",
                "text_field": "content",
                "weight": 0.10,
            },
        ],
    },

    # ------------------------------------------------------------------
    # 7. INFERENCE LOGIC
    # ------------------------------------------------------------------
    "logic": {
        "internal_threshold": 0.5,
        "modes": {
            "instant": {"base": 2, "max": 5},
            "fast": {"base": 3, "max": 8},
            "thinking": {"base": 8, "max": 24},
        },
        "scaling": {
            "length_weight": 0.2,
        },
        "default_steps": 5,
        "min_steps": 3,
        "max_steps": 24,
    },

    # ------------------------------------------------------------------
    # 8. SERVER
    # ------------------------------------------------------------------
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_token": "thunder-secret-at-2026",
    },
    # ------------------------------------------------------------------
    # 9. STORAGE (R2 / S3)
    # ------------------------------------------------------------------
    "storage": {
        "enabled": True,               # Toggle for object storage syncing
        "provider": "r2",
        "bucket": None,                # Loaded from THUNDER_R2_BUCKET
        "endpoint_url": None,          # Loaded from THUNDER_R2_ENDPOINT
        "region": "auto",              # Loaded from THUNDER_R2_REGION
        "access_key_id": None,         # Loaded from THUNDER_R2_ACCESS_KEY
        "secret_access_key": None,     # Loaded from THUNDER_R2_SECRET_KEY
    },
}
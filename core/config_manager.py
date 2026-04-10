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
        "max_seq_len": 8192,           # Increased for Big Run
        "max_gen_len": 1024,
        "device": "auto",
    },

    # ------------------------------------------------------------------
    # 2. FROM-SCRATCH MODEL
    # ------------------------------------------------------------------
    "model": {
        "vocab_size": 49152,           # Locked: SmolLM2-135M tokenizer. Do not change after pretraining starts.
        "embedding_dim": 1536,           # Token space used for clamping and logits.
        "latent_dim": 1280,              # Compressed denoising space for cheaper diffusion steps.
        "ffn_hidden_size": 5120,
        "num_layers": 28,
        "num_attention_heads": 20,
        "num_kv_heads": 5,             # GQA: 4:1 ratio
        "dropout": 0.0,
        "max_seq_len": 8192,
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
        "batch_size": 2,
        "grad_accum": 16,
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
        "fast_steps": 8,
        "thinking_steps": 24,
    },

    # ------------------------------------------------------------------
    # 5. TRAINING
    # ------------------------------------------------------------------
    "training": {
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "warmup_steps": 2000,
        "curriculum_stage_steps": 2500,
        "epochs": 1,
        "max_steps": 250000,           # Full Chinchilla target
        "output_dir": "./runs/thunder_v1_8k_production",
        "save_steps": 500,
        "logging_steps": 1,
        "preview_steps": 100,
        "save_total_limit": 5,
        "t_round_penalty": 0.0,
        "resume_from": None,           # Starting fresh with GQA architecture
        "seed": 3407,
        "pipeline_key": "pretrain_hf_datasets",
        "max_train_blocks": 8000000,   # ~16B tokens target
        "max_documents_per_dataset": 1000000,
    },

    # ------------------------------------------------------------------
    # 6. DATA PIPELINE
    # ------------------------------------------------------------------
    "pipeline": {
        "num_proc": 4,
        "packing": True,
        "block_size": 8192,
        "curriculum_lengths": [1024, 2048, 4096, 8192],
        "eos_between_documents": True,
        "shuffle_seed": 3407,
        "pretrain_hf_datasets": [
            {
                "path": "identity_data/thunder_identity.jsonl",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "weight": 1.0,
            },
            {
                "path": "HuggingFaceTB/cosmopedia-v2",
                "name": "default",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.20,        # High-density reasoning/knowledge
            },
            {
                "path": "HuggingFaceTB/smollm-corpus",
                "name": "fineweb-edu-dedup",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.30,        # High-quality English web
            },
            {
                "path": "readerbench/FuLG",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.30,        # Romanian language core
            },
            {
                "path": "open-web-math/open-web-math",
                "split": "train",
                "format": "text",
                "text_field": "text",
                "streaming": True,
                "weight": 0.10,        # Mathematical reasoning
            },
            {
                "path": "codeparrot/codeparrot-clean",
                "split": "train",
                "format": "text",
                "text_field": "content",
                "weight": 0.10,        # Coding logic
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
            "instant": {"base": 3, "max": 8},
            "fast": {"base": 8, "max": 15},
            "thinking": {"base": 24, "max": 32},
        },
        "scaling": {
            "length_weight": 0.2,
        },
        "default_steps": 8,
        "min_steps": 5,
        "max_steps": 32,
    },

    # ------------------------------------------------------------------
    # 8. SERVER
    # ------------------------------------------------------------------
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_token": "thunder-secret-at-2026",
    },
}

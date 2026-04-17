"""
Thunder dLLM — Pilot Training Run (L40S)
========================================
Lanseaza antrenamentul real pe o placa NVIDIA L40S (48GB).
Include persistenta prin Modal Volumes si monitorizare WandB.

Rulare:
    modal run scripts/modal_train.py
"""

import os
import modal

# Configurare Proiect
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Definim imaginea cu toate dependintele necesare
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_HOME": "/cache/huggingface"})
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "tokenizers>=0.19.0",
        "wandb", # Pentru monitorizare
        "tqdm",
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/thunder",
        ignore=[
            ".git",
            "**/__pycache__",
            ".venv",
            "**/*.pyc",
            "runs/**",
            "data/**", # Dataset-ul se incarca prin streaming
            "**/*.log",
        ],
    )
)

# 2. Cream/Atasam un volum persistent pentru checkpoint-uri
# Acesta va fi montat la /checkpoints in container
volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("thunder-cache", create_if_missing=True)

app = modal.App("thunder-pilot-train")

@app.function(
    image=image,
    gpu="L40S:4", # Requesting 4x L40S GPUs (192GB VRAM total)
    volumes={
        "/checkpoints": volume,
        "/cache": cache_volume,
    },
    secrets=[modal.Secret.from_name("wandb")], 
    timeout=86400, # 24 hours for the Big Run
)
def train():
    import sys
    import os
    
    # Thunder este in /thunder
    os.chdir("/thunder")
    sys.path.insert(0, "/thunder")
    
    from core.config_manager import THUNDER_CONFIG
    from training.diffusion_lm_trainer import run_training
    
    print("⚡ Thunder: Incepem The Big Run pe 4x L40S...")
    
    # Suprascriem configuratia pentru a salva pe volumul persistent
    THUNDER_CONFIG["training"]["output_dir"] = "/checkpoints/thunder_v1_8k_production"
    
    # Asiguram ca folosim precizia optima pentru L40S
    THUNDER_CONFIG["training"]["precision"] = "bf16"
    
    # Activam WandB daca secretul este prezent
    if os.environ.get("WANDB_API_KEY"):
        THUNDER_CONFIG["training"]["use_wandb"] = True
        THUNDER_CONFIG["training"]["wandb_project"] = "thunder-dllm"
    
    # Rulam training-ul principal
    run_training()

@app.local_entrypoint()
def main():
    print("\n🚀 Lansam Pilot Training (3 ore) pe Modal (GPU: L40S)...")
    print("Checkpoint-urile vor fi salvate in volumul: thunder-checkpoints")
    print("Monitorizare: WandB (asigura-te ca ai setat secretul 'wandb' in Modal)\n")
    
    train.remote()

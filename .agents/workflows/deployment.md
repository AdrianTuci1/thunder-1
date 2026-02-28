---
description: Deployment guide for Thunder engine (Installation, Fine-tuning, Execution)
---

# Deployment Workflow: Thunder Engine on VPS / RunPod

This guide covers the steps to set up, fine-tune, and run Thunder on a remote server with NVIDIA GPU support.

## 0. RunPod Pre-Configuration (Recommended)
If you are using RunPod, follow these steps before connecting:

1. **Template**: Choose `RunPod PyTorch` (CUDA 12.1 or 12.4).
2. **GPU**: Select an `RTX 4090` (24GB VRAM) or `A100` for optimal performance.
3. **Expose Ports**:
   - In **TCP Port Mapping**, add Port `8000` (for Thunder WebSocket).
   - In **HTTP Port Mapping**, add Port `8000` (for API access).
4. **Volume**: Ensure at least 50GB of disk space for the model and fine-tuning weights.

## 1. Environment Setup
Connect via SSH and prepare the system.

// turbo
1. Install system dependencies:
```bash
sudo apt-get update && sudo apt-get install -y git python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
```

2. Clone the repository and navigate to it:
```bash
git clone <your-repo-url> thunder && cd thunder
```

3. Initialize the virtual environment and install Python dependencies:
```bash
bash setup_env.sh
```
> [!NOTE]
> `setup_env.sh` installs `torch`, `xformers`, `unsloth`, and `fastapi`.

## 2. Fine-Tuning (SFT)
Configure and run the crystallization tuning.

1. Ensure your dataset is ready in `training/data_pipeline.py`.
2. Run the fine-tuning script:
```bash
python3 training/finetune_gemini.py
```
> [!TIP]
> This will use the settings in `config_manager.py` (LoRA rank, learning rate) to align the model for hierarchical diffusion.

## 3. Execution (Production)
Run the inference engine.

1. Start the FastAPI/WebSocket server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. (Optional) Run in background using `screen` or `pm2`:
```bash
pm2 start "uvicorn app:app --host 0.0.0.0 --port 8000" --name thunder-engine
```

## 4. Verification
Test the connection from your local machine:
```bash
wscat -c ws://<vps-ip>:8000/ws/thunder
```
Send a JSON to test modes:
```json
{"query": "Hello Thunder", "mode": "fast"}
```

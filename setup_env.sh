#!/bin/bash
# setup_env.sh - Environment automation for Thunder framework

echo "Initializing environment for Thunder (CUDA 12.x, Unsloth, SSM)..."

# Install python dependencies
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install fastapi uvicorn websockets torch torchvision torchaudio

echo "Environment initialization complete. ✅ Thunder Ready."

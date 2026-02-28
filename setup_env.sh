#!/bin/bash
# setup_env.sh - Environment automation for Thunder framework

echo "Initializing isolated environment for Thunder (CUDA 12.x, Unsloth)..."

# Create and activate virtual environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install python dependencies in the isolated environment
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install fastapi uvicorn websockets

echo "Environment initialization complete. ✅ Thunder Ready (Isolated)."

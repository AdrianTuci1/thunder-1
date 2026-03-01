#!/bin/bash

# Configuration
LOG_FILE="training.log"
VENV_PATH="venv"
PYTHON_BIN="$VENV_PATH/bin/python3"

# Environment Setup
echo "⚡ Thunder Startup: Configuring environment..."
mkdir -p /tmp/hf_cache /tmp/torch_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export BNB_CUDA_VERSION=128

# Clean old log
> "$LOG_FILE"

echo "⚡ Thunder Startup: Starting training in background (unbuffered)..."
nohup "$PYTHON_BIN" -u training/diffusion_lm_trainer.py >> "$LOG_FILE" 2>&1 &

PID=$!
echo "⚡ Thunder Startup: Process started with PID: $PID"
echo "⚡ Thunder Startup: Following logs..."
echo "--------------------------------------------------"
tail -f "$LOG_FILE"

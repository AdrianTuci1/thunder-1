#!/bin/bash
# start_training.sh - Helper script for Thunder dLLM training

# Set higher file limits for massive dataset streaming
ulimit -n 65536
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Change to the root of the thunder project
cd "$(dirname "$0")/.."

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "⚙️ Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️ Warning: .env file not found."
fi

# Ensure output directory exists (mapped to config output_dir)
mkdir -p ./runs/thunder_v1_850M_production

# Print hardware info
echo "🚀 Starting Thunder Training on: $(nvidia-smi -L | head -n 1)"
echo "📌 Logging to project: thunder-dllm"

# Launch training with accelerate
accelerate launch training/diffusion_lm_trainer.py

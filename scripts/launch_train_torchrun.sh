#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_SCRIPT:?Set TRAIN_SCRIPT to the training entrypoint, e.g. training/run_from_scratch.py}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "== torchrun launcher =="
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "NNODES=$NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

exec torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  "$@"

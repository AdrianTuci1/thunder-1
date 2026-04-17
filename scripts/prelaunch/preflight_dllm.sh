#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_MANIFEST="$ROOT_DIR/data/manifests/dllm_corpus_manifest.local.json"
EXAMPLE_MANIFEST="$ROOT_DIR/data/manifests/dllm_corpus_manifest.example.json"

echo "== Thunder dLLM preflight =="
python3 "$ROOT_DIR/scripts/audit_training_readiness.py"

if [[ -f "$LOCAL_MANIFEST" ]]; then
  python3 "$ROOT_DIR/scripts/verify_dataset_integrity.py" --manifest "$LOCAL_MANIFEST"
else
  echo
  echo "Local manifest not found, checking the example manifest instead."
  python3 "$ROOT_DIR/scripts/verify_dataset_integrity.py" --manifest "$EXAMPLE_MANIFEST"
fi

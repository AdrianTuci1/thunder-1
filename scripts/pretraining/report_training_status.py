#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_checkpoint(checkpoint_root: Path):
    checkpoints = sorted(
        [path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()],
        key=lambda item: int(item.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None, checkpoints


def main():
    parser = argparse.ArgumentParser(description="Report the latest local training status.")
    parser.add_argument("--run-dir", default="thunder_qwen_32k", help="Run directory that contains metrics and checkpoints.")
    parser.add_argument("--metrics", default=None, help="Optional explicit path to metrics.jsonl.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "metrics.jsonl"
    checkpoint_root = run_dir

    metrics = load_metrics(metrics_path)
    last_checkpoint, all_checkpoints = latest_checkpoint(checkpoint_root)

    print(f"Run directory: {run_dir.resolve()}")
    print(f"Metrics file: {metrics_path.resolve()}")
    print(f"Checkpoints found: {len(all_checkpoints)}")
    if last_checkpoint:
        print(f"Latest checkpoint: {last_checkpoint.name}")
    else:
        print("Latest checkpoint: none")

    if not metrics:
        print("No metrics recorded yet.")
        return

    last_row = metrics[-1]
    best_row = min(metrics, key=lambda row: row.get("loss", float("inf")))

    print()
    print(f"Latest step: {last_row.get('step')}")
    print(f"Latest epoch: {last_row.get('epoch')}")
    print(f"Latest loss: {last_row.get('loss')}")
    print(f"Latest denoising loss: {last_row.get('denoising_loss')}")
    print(f"Latest learning rate: {last_row.get('learning_rate')}")
    print(f"Latest grad norm: {last_row.get('grad_norm')}")
    print(f"Best loss so far: {best_row.get('loss')} at step {best_row.get('step')}")


if __name__ == "__main__":
    main()

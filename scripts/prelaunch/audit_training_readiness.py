#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def has_text(relative_path: str, pattern: str) -> bool:
    return pattern in read_text(relative_path)


def repo_has_pattern(pattern: str) -> bool:
    regex = re.compile(pattern)
    for path in ROOT.rglob("*.py"):
        if path.name == "audit_training_readiness.py":
            continue
        if regex.search(path.read_text(encoding="utf-8")):
            return True
    return False


def build_checks():
    checks = []

    model_loader_text = read_text("core/model_loader.py")
    config_text = read_text("core/config_manager.py")
    trainer_text = read_text("training/diffusion_lm_trainer.py")
    data_pipeline_text = read_text("training/data_pipeline.py")
    scratch_model_text = read_text("core/scratch_dllm.py")
    diffusion_engine_text = read_text("core/diffusion_engine.py")
    blueprint = json.loads(read_text("configs/dllm_1b_blueprint.json"))

    from_scratch_ready = (
        exists("core/scratch_dllm.py")
        and exists("training/run_from_scratch.py")
        and '"model_source": "scratch"' in config_text
        and "ThunderScratchDiffusionLM" in model_loader_text
    )
    checks.append(
        {
            "name": "From-scratch 0-1B training",
            "status": "OK" if from_scratch_ready else "MISSING",
            "detail": "A scratch model, scratch training entrypoint and scratch loader path should all exist together.",
        }
    )

    target_scope_ok = (
        blueprint["model"].get("max_seq_len") == 2048
        and 800_000_000 <= blueprint["model"].get("target_params", 0) <= 1_000_000_000
    )
    checks.append(
        {
            "name": "Frozen project scope",
            "status": "OK" if target_scope_ok else "PARTIAL",
            "detail": "Blueprint should stay focused on ~0.8B-1B parameters and 2048 context so the project does not drift toward unnecessary long-context work.",
        }
    )

    has_bidirectional_switch = (
        "build_bidirectional_attention_mask" in scratch_model_text
        and "is_causal=False" in scratch_model_text
        and exists("tests/test_scratch_dllm.py")
    )
    checks.append(
        {
            "name": "Bidirectional attention / no causal mask",
            "status": "OK" if has_bidirectional_switch else "MISSING",
            "detail": "Scratch attention should use non-causal SDPA plus an explicit unit test for the bidirectional mask.",
        }
    )

    has_dynamic_canvas = "max_new_tokens" in diffusion_engine_text and "Dynamic Canvas" in diffusion_engine_text
    canvas_in_training = "_apply_length_curriculum" in trainer_text and "curriculum_lengths" in config_text
    checks.append(
        {
            "name": "Dynamic canvas / length curriculum",
            "status": "OK" if has_dynamic_canvas and canvas_in_training else ("PARTIAL" if has_dynamic_canvas else "MISSING"),
            "detail": "Inference should expose dynamic output canvas, and training should stage sequence lengths instead of jumping straight to 2048 everywhere.",
        }
    )

    has_parallel_blocks = "pack_texts" in data_pipeline_text and "block_size" in data_pipeline_text
    strong_block_pipeline = has_parallel_blocks and exists("tests/test_data_pipeline.py")
    checks.append(
        {
            "name": "Packed text block processing",
            "status": "OK" if strong_block_pipeline else ("PARTIAL" if has_parallel_blocks else "MISSING"),
            "detail": "The repo should expose a fixed-length packer for 2048-token blocks and cover it with a unit test.",
        }
    )

    manifest_local = exists("data/manifests/dllm_corpus_manifest.local.json")
    raw_files = [path for path in (ROOT / "data").rglob("*") if path.is_file() and "manifest" not in path.parts]
    hf_verifier = exists("scripts/verify_hf_dataset_sources.py") and "pretrain_hf_datasets" in config_text
    checks.append(
        {
            "name": "Dataset integrity workflow",
            "status": "OK" if manifest_local and raw_files else ("PARTIAL" if hf_verifier else "MISSING"),
            "detail": (
                "Local manifest and raw shards are present."
                if manifest_local and raw_files
                else "Hugging Face source verification is wired in, but a populated local manifest with real downloaded shards is still missing."
                if hf_verifier
                else "A manifest example exists, but there is no populated local manifest with real dataset shards checked into the workspace."
            ),
        }
    )

    launcher_candidates = [
        path
        for path in (ROOT / "scripts").glob("*")
        if path.is_file() and ("launch" in path.name or "torchrun" in path.name or "cluster" in path.name)
    ]
    distributed_keywords = bool(launcher_candidates)
    checks.append(
        {
            "name": "Distributed cluster launcher",
            "status": "OK" if distributed_keywords else "MISSING",
            "detail": (
                "Launcher candidates: "
                + ", ".join(path.name for path in launcher_candidates)
                if distributed_keywords
                else "No dedicated multi-node launch entrypoint or distributed training config was found in the repo."
            ),
        }
    )

    has_resume = "optimizer.pt" in trainer_text and "scheduler.pt" in trainer_text and "trainer_state.json" in trainer_text
    checks.append(
        {
            "name": "Checkpoint resume",
            "status": "OK" if has_resume else "MISSING",
            "detail": "Trainer now saves optimizer, scheduler and trainer state, but the distributed launcher still needs to restore them automatically.",
        }
    )

    has_metrics = "metrics.jsonl" in trainer_text and exists("scripts/report_training_status.py")
    checks.append(
        {
            "name": "Structured monitoring",
            "status": "OK" if has_metrics else "MISSING",
            "detail": "Metrics JSONL logging is present. External dashboards and alerts are still recommended for long runs.",
        }
    )

    stale_refs = []
    if not exists("core/tile_manager.py") and repo_has_pattern(r"core\.tile_manager"):
        stale_refs.append("core.tile_manager")
    if "PrefixLMDiffusionEngine" in read_text("app.py"):
        stale_refs.append("PrefixLMDiffusionEngine")
    checks.append(
        {
            "name": "Repo consistency",
            "status": "PARTIAL" if stale_refs else "OK",
            "detail": "Stale references found: " + ", ".join(stale_refs) if stale_refs else "No obvious stale module references were detected by the lightweight audit.",
        }
    )

    return checks


def print_human(checks):
    print("Thunder dLLM readiness audit")
    print("=" * 32)
    for check in checks:
        print(f"[{check['status']}] {check['name']}")
        print(f"  {check['detail']}")

    missing = sum(1 for check in checks if check["status"] == "MISSING")
    partial = sum(1 for check in checks if check["status"] == "PARTIAL")
    print()
    print(f"Summary: {missing} missing, {partial} partial, {len(checks) - missing - partial} ok.")


def main():
    parser = argparse.ArgumentParser(description="Audit the repo for from-scratch dLLM training readiness.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any check is missing.")
    args = parser.parse_args()

    checks = build_checks()
    if args.json:
        print(json.dumps({"checks": checks}, indent=2))
    else:
        print_human(checks)

    if args.strict and any(check["status"] == "MISSING" for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

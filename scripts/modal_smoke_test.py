"""
Thunder dLLM — Modal Smoke Test (cu codul real montat)
======================================================
Monteaza intregul proiect Thunder in container si importa direct
din core/ si training/ — testeaza codul real, nu o copie inline.

Rulare:
    modal run scripts/modal_smoke_test.py

Optiuni:
    modal run --detach scripts/modal_smoke_test.py
    modal app logs thunder-smoke-test

Cost estimat: < $0.05  (T4, ~3-5 minute incluzand cold start si downloads)
GPU ales: T4 (16GB VRAM) — suficient pentru config-ul de test redus.
"""

import os
import modal

# ---------------------------------------------------------------------------
# Proiectul Thunder este montat in container la /thunder
# Modal urca automat tot ce e in directorul local la rularea comenzii.
# ---------------------------------------------------------------------------

# PROJECT_ROOT este folosit mai jos in definitia imaginii.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "tokenizers>=0.19.0",
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
            "data/**",
            "**/*.pdf",
            "**/*.log",
            "inference_test.log",
        ],
    )
)

app = modal.App("thunder-smoke-test")

# ---------------------------------------------------------------------------
# Config smoke test — model mai mic decat cel de productie, pentru viteza
# ---------------------------------------------------------------------------

SMOKE_OVERRIDES = {
    # Dimensiuni reduse fata de productie (28L/1280D) — doar pentru smoke
    "num_layers": 4,
    "embedding_dim": 256,
    "latent_dim": 256,
    "ffn_hidden_size": 512,
    "num_attention_heads": 8,
    "num_kv_heads": 2,       # Missing GQA parameter
    "max_seq_len": 128,      # bloc mic, nu 8192
}

SMOKE_DIFFUSION_STEPS = 16
SMOKE_BATCH_SIZE = 2
SMOKE_N_DOCUMENTS = 60      # documente din HF streaming
SMOKE_MAX_BLOCKS = 8        # blocuri packed maxim
SMOKE_DATASET_SPEC = {      # doar primul dataset din mix, pentru viteza
    "path": "HuggingFaceTB/smollm-corpus",
    "name": "fineweb-edu-dedup",
    "split": "train",
    "format": "text",
    "text_field": "text",
    "streaming": True,
    "max_documents": SMOKE_N_DOCUMENTS,
}


# ---------------------------------------------------------------------------
# Functia principala
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100",
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=600,
)
def run_smoke_test():
    import sys
    import time
    import torch
    import torch.nn.functional as F

    # Thunder este montat la /thunder — il adaugam in Python path
    sys.path.insert(0, "/thunder")

    from core.config_manager import THUNDER_CONFIG
    from core.scratch_dllm import ScratchDLMConfig, ThunderScratchDiffusionLM
    from training.data_pipeline import ThunderDataPipeline
    from transformers import AutoTokenizer

    results: dict[str, bool] = {}
    t_start = time.time()

    def ok(label: str, value=True):
        results[label] = bool(value)
        icon = "✅" if value else "❌"
        print(f"  {icon} {label}")

    def section(title: str):
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    print("\n" + "=" * 60)
    print("  Thunder dLLM — Smoke Test (cod real montat)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. GPU / CUDA
    # ------------------------------------------------------------------
    section("1. Hardware")
    cuda_ok = torch.cuda.is_available()
    ok("CUDA disponibil", cuda_ok)
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"     GPU: {device_name}")
        print(f"     VRAM: {vram:.1f} GB")
        ok(f"GPU detectat ({device_name})")
    else:
        print("  ATENTIE: CUDA nu e disponibil, testul continua pe CPU")

    device = torch.device("cuda" if cuda_ok else "cpu")

    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        # T4 raporteaza eronat suport bf16 uneori, asa ca verificam si numele placii
        if torch.cuda.is_bf16_supported() and "T4" not in device_name:
            dtype = torch.bfloat16
            print(f"     Dtype: bfloat16 (hardware supported on {device_name})")
        else:
            dtype = torch.float16
            print(f"     Dtype: float16 (T4 or non-ampere fallback)")
    else:
        dtype = torch.float32
        print("     Dtype: float32 (CPU)")

    # ------------------------------------------------------------------
    # 2. Config + Tokenizer din THUNDER_CONFIG
    # ------------------------------------------------------------------
    section("2. Tokenizer — din THUNDER_CONFIG")
    tokenizer = None
    try:
        tokenizer_name = THUNDER_CONFIG["engine"]["tokenizer_name"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = len(tokenizer)
        ok(f"Tokenizer incarcat: {tokenizer_name}")
        ok(f"vocab_size == 49152", vocab_size == 49152)
        print(f"     vocab_size: {vocab_size}")
        print(f"     eos_token_id: {tokenizer.eos_token_id}")

        sample = "Diffusion models denoise tokens in parallel using bidirectional attention."
        ids = tokenizer.encode(sample)
        ok("Tokenizare text simplu", len(ids) > 0)
        print(f"     Sample ({len(sample.split())} cuvinte → {len(ids)} tokens): {ids[:10]}...")

    except Exception as exc:
        ok("Tokenizer incarcat", False)
        print(f"     EROARE: {exc}")

    if tokenizer is None:
        _print_summary(results, t_start)
        return results

    # ------------------------------------------------------------------
    # 3. Dataset via ThunderDataPipeline (codul real)
    # ------------------------------------------------------------------
    section("3. Dataset via ThunderDataPipeline")
    dataset = None
    try:
        # Suprascriem block_size in config-ul global pentru a se alinia cu modelul de smoke test
        THUNDER_CONFIG["pipeline"]["block_size"] = SMOKE_OVERRIDES["max_seq_len"]

        pipeline = ThunderDataPipeline(
            tokenizer=tokenizer,
            max_seq_length=SMOKE_OVERRIDES["max_seq_len"],
        )
        dataset = pipeline.prepare_dataset(
            dataset_specs=[SMOKE_DATASET_SPEC],
            max_blocks=SMOKE_MAX_BLOCKS,
            max_documents_per_dataset=SMOKE_N_DOCUMENTS,
        )
        ok("ThunderDataPipeline.prepare_dataset() fara erori")
        ok(f"Dataset instantiat corect", dataset is not None)
        # StreamingBlockDataset nu are len(), luam un exemplu dintr-un iterator scurt
        sample_batch = next(iter(dataset))
        ok("Dataset produce date valide", "input_ids" in sample_batch)
        print(f"     Primul block (primii 10 ids): {sample_batch['input_ids'][:10]}")

    except Exception as exc:
        ok("ThunderDataPipeline.prepare_dataset() fara erori", False)
        print(f"     EROARE: {exc}")

    # ------------------------------------------------------------------
    # 4. Model from-scratch cu ScratchDLMConfig
    # ------------------------------------------------------------------
    section("4. Model — ThunderScratchDiffusionLM (config smoke)")
    model = None
    try:
        cfg_dict = {**THUNDER_CONFIG["model"], **SMOKE_OVERRIDES}
        cfg_dict["vocab_size"] = len(tokenizer)

        model_cfg = ScratchDLMConfig(
            vocab_size=cfg_dict["vocab_size"],
            embedding_dim=cfg_dict["embedding_dim"],
            latent_dim=cfg_dict["latent_dim"],
            ffn_hidden_size=cfg_dict["ffn_hidden_size"],
            num_layers=cfg_dict["num_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_kv_heads=cfg_dict["num_kv_heads"], # Added GQA support
            max_seq_len=cfg_dict["max_seq_len"],
            pad_token_id=THUNDER_CONFIG["model"].get("pad_token_id", 0),
            self_conditioning=THUNDER_CONFIG["model"].get("self_conditioning", True),
            use_fp8=True, # Test FP8 path
        )

        model = ThunderScratchDiffusionLM(
            config=model_cfg,
            diffusion_steps=SMOKE_DIFFUSION_STEPS,
        ).to(device=device, dtype=dtype)

        n_params = model.num_parameters()
        estimated = model_cfg.estimate_parameter_count()
        ok("ThunderScratchDiffusionLM instantiat fara erori")
        ok(f"vocab_size in model == {len(tokenizer)}",
           model.config.vocab_size == len(tokenizer))
        print(f"     Parametri reali smoke config: {n_params:,}")
        print(f"     Parametri estimati config: {estimated:,}")
        print(f"     Config: {model_cfg.num_layers}L / {model_cfg.latent_dim}D / "
              f"{model_cfg.num_attention_heads}H / seq={model_cfg.max_seq_len}")

    except Exception as exc:
        ok("ThunderScratchDiffusionLM instantiat fara erori", False)
        print(f"     EROARE: {exc}")
        _print_summary(results, t_start)
        return results

    # ------------------------------------------------------------------
    # 5. Forward pass pe date reale (sau random dacă dataset a eșuat)
    # ------------------------------------------------------------------
    section("5. Forward pass (date reale / bidirectional)")
    loss = None
    try:
        if dataset is not None:
            # Luam primul batch din iteratorul de streaming
            batch = next(iter(dataset))
            block_ids = batch["input_ids"]
            if isinstance(block_ids, list):
                block_ids = torch.tensor(block_ids)
            src = "date reale din flux"
        else:
            block_ids = torch.randint(0, len(tokenizer), (SMOKE_OVERRIDES["max_seq_len"],))
            src = "date random (fallback)"

        input_ids = block_ids.unsqueeze(0).expand(SMOKE_BATCH_SIZE, -1).to(device)
        timesteps = torch.randint(
            0, SMOKE_DIFFUSION_STEPS, (SMOKE_BATCH_SIZE,), device=device
        )

        model.eval()
        with torch.no_grad(), torch.autocast(
            device_type="cuda" if cuda_ok else "cpu", dtype=dtype
        ):
            # ThunderScratchDiffusionLM.forward() → x0_pred in embedding space
            x0_pred = model(input_ids, timesteps)

        seq_len = SMOKE_OVERRIDES["max_seq_len"]
        expected_shape = (SMOKE_BATCH_SIZE, seq_len, model_cfg.embedding_dim)
        ok(f"Forward pass fara erori ({src})")
        ok(f"Shape x0_pred corect {expected_shape}", tuple(x0_pred.shape) == expected_shape)
        print(f"     x0_pred shape: {tuple(x0_pred.shape)}")

        # Logits prin dot-product cu embedding table (token clamping)
        embed_weight = model.token_embeddings.weight  # [V, E]
        logits = x0_pred.float() @ embed_weight.float().T  # [B, L, V]
        targets = input_ids.long()
        loss = F.cross_entropy(
            logits.reshape(-1, len(tokenizer)),
            targets.reshape(-1),
        )
        loss_val = loss.item()
        import math
        expected_loss = math.log(len(tokenizer))
        ok("Loss calculat fara NaN/Inf",
           not (loss_val != loss_val or loss_val == float("inf")))
        print(f"     Loss: {loss_val:.4f}  "
              f"(random init ~ ln({len(tokenizer)}) = {expected_loss:.2f})")
        ok("Loss in range rezonabil (< 15.0)", loss_val < 15.0)

    except Exception as exc:
        ok("Forward pass fara erori", False)
        print(f"     EROARE: {exc}")
        _print_summary(results, t_start)
        return results

    # ------------------------------------------------------------------
    # 6. Backward pass
    # ------------------------------------------------------------------
    section("6. Backward pass + gradient check")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if cuda_ok else "cpu", dtype=dtype):
            x0_train = model(input_ids, timesteps)

        embed_weight = model.token_embeddings.weight
        logits_train = x0_train.float() @ embed_weight.float().T
        loss_train = F.cross_entropy(
            logits_train.reshape(-1, len(tokenizer)),
            targets.reshape(-1),
        )
        loss_train.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ok("Backward pass fara erori")
        ok("Gradient norm > 0", grad_norm.item() > 0.0)
        print(f"     Gradient norm (post-clip): {grad_norm.item():.4f}")

    except Exception as exc:
        ok("Backward pass fara erori", False)
        print(f"     EROARE: {exc}")

    # ------------------------------------------------------------------
    # Rezumat final
    # ------------------------------------------------------------------
    _print_summary(results, t_start)
    return results


def _print_summary(results: dict, t_start: float):
    import time
    elapsed = time.time() - t_start
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    all_ok = passed == total

    print("\n" + "=" * 60)
    print(f"  SMOKE TEST {'PASSED ✅' if all_ok else 'FAILED ❌'} — {passed}/{total} checks")
    print(f"  Durata totala: {elapsed:.1f}s")
    print("=" * 60)

    if not all_ok:
        print("\n  Checks esuate:")
        for label, ok_val in results.items():
            if not ok_val:
                print(f"    ❌  {label}")
    print()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """
    Urca codul Thunder pe Modal si ruleaza smoke test pe GPU T4.

    Comanda:
        modal run scripts/modal_smoke_test.py

    Async (trimite si nu asteapta):
        modal run --detach scripts/modal_smoke_test.py
        modal app logs thunder-smoke-test
    """
    print("\nUrcam codul Thunder pe Modal si pornim smoke test (GPU: T4)...")
    print("Upload cod + cold start: ~45-90s | Executie: ~2-3min\n")

    results = run_smoke_test.remote()

    if not results:
        print("Smoke test nu a returnat rezultate.")
        return

    all_passed = all(results.values())
    if all_passed:
        print("\n✅  Smoke test complet cu succes.")
        print("    Codul real Thunder functioneaza in container Modal.")
        print("    Tokenizer SmolLM2-135M confirmat (49152 vocab).")
        print("    Pipeline dataset → packing → forward → backward: OK.")
        print("\n    Poti trece la primul run de training real.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n❌  Smoke test a detectat {len(failed)} probleme:")
        for f in failed:
            print(f"    - {f}")
        print("\n    Verifica log-urile de mai sus pentru detalii.")

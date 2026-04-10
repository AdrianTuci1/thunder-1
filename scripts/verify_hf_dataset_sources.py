#!/usr/bin/env python3
import argparse
import json

from core.config_manager import THUNDER_CONFIG


def extract_text(example, spec):
    fmt = spec.get("format", "text")

    if fmt == "text":
        value = example.get(spec.get("text_field", "text"), "")
        return value if isinstance(value, str) else ""

    if fmt in {"conversations", "messages"}:
        messages = example.get(spec.get("messages_field", "messages"), [])
        parts = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role", message.get("from", "user"))
            content = message.get("content", message.get("value", ""))
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    if fmt == "prompt_response":
        prompt = example.get(spec.get("prompt_field", "prompt"), "")
        response = example.get(spec.get("response_field", "response"), "")
        return f"{prompt}\n{response}"

    return ""


def verify_dataset_spec(spec):
    from datasets import load_dataset, load_dataset_builder

    builder = load_dataset_builder(spec["path"], spec.get("name"))
    info = builder.info
    split = spec.get("split", "train")

    split_exists = split in info.splits if info.splits else False
    preview_stream = load_dataset(spec["path"], spec.get("name"), split=split, streaming=True)
    preview_example = next(iter(preview_stream))
    preview_text = extract_text(preview_example, spec)[:160]

    return {
        "dataset": spec["path"],
        "config": spec.get("name"),
        "split": split,
        "license": info.license,
        "split_exists": split_exists,
        "features": list(info.features.keys()) if info.features else [],
        "preview_text": preview_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Hugging Face dataset sources referenced by Thunder config.")
    parser.add_argument(
        "--pipeline-key",
        default="pretrain_hf_datasets",
        choices=["pretrain_hf_datasets", "sft_hf_datasets"],
        help="Which dataset list from THUNDER_CONFIG['pipeline'] to validate.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    specs = THUNDER_CONFIG["pipeline"].get(args.pipeline_key, [])
    results = [verify_dataset_spec(spec) for spec in specs]

    if args.json:
        print(json.dumps({"pipeline_key": args.pipeline_key, "results": results}, indent=2))
        return

    print(f"Hugging Face dataset verification: {args.pipeline_key}")
    print("=" * 44)
    for item in results:
        print(f"[{item['dataset']}] split={item['split']} split_exists={item['split_exists']}")
        print(f"  config={item['config']}")
        print(f"  license={item['license']}")
        print(f"  features={', '.join(item['features'][:8])}")
        print(f"  preview={item['preview_text']}")


if __name__ == "__main__":
    main()

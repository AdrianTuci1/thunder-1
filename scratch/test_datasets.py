from datasets import load_dataset
import os

specs = [
    {"path": "HuggingFaceTB/cosmopedia-v2", "name": "default"},
    {"path": "HuggingFaceTB/smollm-corpus", "name": "fineweb-edu-dedup"},
    {"path": "readerbench/FuLG"},
    {"path": "open-web-math/open-web-math"},
    {"path": "codeparrot/codeparrot-clean"},
]

for spec in specs:
    path = spec["path"]
    name = spec.get("name")
    print(f"Testing {path} (name={name})...")
    try:
        ds = load_dataset(path, name, split="train", streaming=True)
        print(f"✅ Success: {path}")
    except Exception as e:
        print(f"❌ Error for {path}: {e}")

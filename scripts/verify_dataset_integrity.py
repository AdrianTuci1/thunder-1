#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_VALUES = {"", "REPLACE_ME", None}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_text_file(path: Path, compression: str):
    if compression == "gzip" or path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def validate_schema(dataset, path: Path):
    schema = dataset.get("schema", {})
    required_keys = schema.get("required_keys", [])
    sample_records = int(dataset.get("sample_records", 0))
    file_format = dataset.get("format", "")
    compression = dataset.get("compression", "")

    if not required_keys or sample_records <= 0:
        return []

    if file_format not in {"jsonl", "json", "csv"}:
        return [f"schema sampling skipped for unsupported format '{file_format}'"]

    issues = []
    with open_text_file(path, compression) as handle:
        if file_format == "jsonl":
            for index, line in enumerate(handle):
                if index >= sample_records:
                    break
                payload = json.loads(line)
                missing_keys = [key for key in required_keys if key not in payload]
                if missing_keys:
                    issues.append(f"record {index + 1} missing keys: {', '.join(missing_keys)}")
        elif file_format == "json":
            payload = json.load(handle)
            records = payload if isinstance(payload, list) else [payload]
            for index, item in enumerate(records[:sample_records]):
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    issues.append(f"record {index + 1} missing keys: {', '.join(missing_keys)}")
        elif file_format == "csv":
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if index >= sample_records:
                    break
                missing_keys = [key for key in required_keys if key not in row]
                if missing_keys:
                    issues.append(f"record {index + 1} missing columns: {', '.join(missing_keys)}")

    return issues


def validate_dataset(dataset):
    issues = []
    path = resolve_path(dataset["path"])

    if not path.exists():
        issues.append(f"missing file: {path}")
        return issues

    size = path.stat().st_size
    min_bytes = int(dataset.get("min_bytes", 1))
    if size < min_bytes:
        issues.append(f"file too small: {size} bytes < {min_bytes}")

    expected_sha256 = dataset.get("sha256")
    if expected_sha256 not in PLACEHOLDER_VALUES:
        actual_sha256 = sha256sum(path)
        if actual_sha256 != expected_sha256:
            issues.append(f"sha256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    else:
        issues.append("sha256 placeholder not replaced")

    license_name = dataset.get("license")
    if license_name in PLACEHOLDER_VALUES:
        issues.append("license placeholder not replaced")

    issues.extend(validate_schema(dataset, path))
    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify dataset integrity from a local manifest.")
    parser.add_argument(
        "--manifest",
        default="data/manifests/dllm_corpus_manifest.example.json",
        help="Path to the dataset manifest JSON file.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any issue is found.")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])

    print(f"Dataset manifest: {manifest_path}")
    print(f"Datasets declared: {len(datasets)}")

    total_issues = 0
    if not datasets:
        print("No datasets registered in the manifest.")
        total_issues += 1

    for dataset in datasets:
        print(f"\n[{dataset['name']}]")
        issues = validate_dataset(dataset)
        if issues:
            total_issues += len(issues)
            for issue in issues:
                print(f"- {issue}")
        else:
            print("- ok")

    print()
    print(f"Integrity summary: {total_issues} issue(s).")

    if args.strict and total_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

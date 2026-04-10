import sys
import os
from pathlib import Path

# Adaugam proiectul in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config_manager import THUNDER_CONFIG
from training.data_pipeline import ThunderDataPipeline
from transformers import AutoTokenizer

def test_load():
    print("Testing sequence...")
    print(f"Dataset path: {THUNDER_CONFIG['pipeline']['pretrain_hf_datasets'][0]['path']}")
    
    # Folosim tokenizer-ul configurat
    tokenizer = AutoTokenizer.from_pretrained(THUNDER_CONFIG["engine"]["tokenizer_name"])
    pipeline = ThunderDataPipeline(tokenizer)
    
    # Incarcam doar dataset-ul de identitate pentru test
    identity_spec = THUNDER_CONFIG["pipeline"]["pretrain_hf_datasets"][0]
    
    print(f"Loading identity dataset from: {identity_spec['path']}")
    dataset = pipeline.prepare_dataset(
        dataset_specs=[identity_spec],
        max_documents_per_dataset=100
    )
    
    print(f"Successfully loaded {len(dataset)} blocks.")
    if len(dataset) > 0:
        # Decodam primul bloc sa vedem ce e in el
        sample = dataset[0]["input_ids"]
        decoded = tokenizer.decode(sample)
        print("\nSample decoded content:")
        print(decoded[:500] + "...")
        
        if "Thunder 1" in decoded:
            print("\n✅ Verification SUCCESS: 'Thunder 1' found in dataset!")
        else:
            print("\n❌ Verification FAILED: 'Thunder 1' NOT found in dataset.")
    else:
        print("\n❌ Verification FAILED: Dataset is empty.")

if __name__ == "__main__":
    test_load()

from datasets import load_dataset
from core.config_manager import THUNDER_CONFIG

class ThunderDataPipeline:
    """
    Handles data loading and preprocessing for 120k context windows.
    Implements Constant Length Packing to ensure maximum GPU utilization.
    """
    
    def __init__(self, tokenizer, max_seq_length=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length or THUNDER_CONFIG["engine"]["max_seq_len"]

    def prepare_dataset(self, dataset_name=None):
        """
        Loads the dataset automatically from HuggingFace and applies Constant Length Packing.
        Downloads and caches on the first run.
        """
        dataset_name = dataset_name or THUNDER_CONFIG["training"]["dataset_name"]
        dataset = load_dataset(dataset_name, split="train")
        
        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.max_seq_length,
                return_overflowing_tokens=False,
            )
            return {"input_ids": outputs["input_ids"]}

        # Apply tokenization and packing logic
        # Constant Length Packing is handled by SFTTrainer, but we prepare the mapping here
        print(f"⚡ Thunder: Mapping tokens for {self.max_seq_length} context...")
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        
        return dataset

    def shard_for_fractal(self, tokens, meso_size=16384, micro_size=2048):
        """
        Splits tokens into the Macro/Meso/Micro hierarchy for training.
        Ensures temporal consistency for the bidirectional denoising loss.
        """
        total_len = len(tokens)
        shards = []
        
        for i in range(0, total_len, meso_size):
            meso_shard = tokens[i : i + meso_size]
            micro_shards = [
                meso_shard[j : j + micro_size] 
                for j in range(0, len(meso_shard), micro_size)
            ]
            shards.append({
                "meso_index": i // meso_size,
                "micro_shards": micro_shards
            })
            
        return shards

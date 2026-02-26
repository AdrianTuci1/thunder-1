import torch
from datasets import load_dataset
from core.config_manager import THUNDER_CONFIG
from training.noise_scheduler import ThunderNoiseScheduler

class ThunderDataPipeline:
    """
    Handles data loading and preprocessing for 120k context windows.
    Implements Constant Length Packing to ensure maximum GPU utilization.
    Now supports Noise Augmentation for Diffusion-based training.
    """
    
    def __init__(self, tokenizer, max_seq_length=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length or THUNDER_CONFIG["engine"]["max_seq_len"]
        self.noise_scheduler = ThunderNoiseScheduler()

    def prepare_dataset(self, dataset_names=None, augment=True):
        """
        Loads multiple datasets and interleaves them based on configured ratios.
        Supports automatic alignment of different data structures (e.g. SlimOrca vs UltraFeedback).
        """
        from datasets import interleave_datasets, load_dataset
        
        dataset_names = dataset_names or THUNDER_CONFIG["training"]["dataset_name"]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
            
        ratios = THUNDER_CONFIG["training"].get("dataset_ratios", [1.0 / len(dataset_names)] * len(dataset_names))
        
        loaded_datasets = []
        for name in dataset_names:
            print(f"⚡ Thunder: Loading dataset {name}...")
            # Load dataset (SlimOrca is large, we use streaming or just the train split)
            ds = load_dataset(name, split="train")
            
            # 1. Normalize columns to a common "text" format if needed
            # UltraFeedback and SlimOrca have different structures
            ds = self._normalize_dataset(ds, name)
            loaded_datasets.append(ds)

        # 2. Interleave
        print(f"⚡ Thunder: Interleaving {len(loaded_datasets)} datasets with ratios {ratios}...")
        dataset = interleave_datasets(loaded_datasets, probabilities=ratios, seed=THUNDER_CONFIG["training"]["seed"])
        
        def process_element(element):
            tokens = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"]
            
            if augment:
                # 1. Randomly sample timesteps for noise augmentation
                # We simulate different stages of the diffusion process
                t = torch.randint(0, self.noise_scheduler.num_train_timesteps, (1,)).item()
                
                # 2. In a real diffusion setup, we'd apply noise to embeddings/latents.
                # Here we mark the element with the target noise level for the trainer.
                # The actual noise injection will happen in the training loop/collator.
                return {
                    "input_ids": input_ids.flatten().tolist(),
                    "diffusion_step": t,
                    "is_augmented": True
                }
            
            return {"input_ids": input_ids.flatten().tolist()}

        print(f"⚡ Thunder: Preparing dataset (Augmentation: {augment})...")
        dataset = dataset.map(process_element, batched=False, remove_columns=dataset.column_names)
        
        return dataset

    def _normalize_dataset(self, dataset, name):
        """
        Standardizes different dataset formats into a unified 'text' column.
        """
        if "SlimOrca" in name:
            # SlimOrca uses a list of messages
            def format_orca(example):
                chat = example["conversations"]
                text = ""
                for msg in chat:
                    role = "User" if msg["from"] == "human" else "Assistant"
                    text += f"### {role}:\n{msg['value']}\n\n"
                return {"text": text}
            return dataset.map(format_orca, remove_columns=dataset.column_names)
            
        elif "LongAlign" in name:
            # LongAlign usually has instruction/output or a flat structure
            def format_long(example):
                return {"text": f"### User:\n{example.get('instruction', example.get('prompt', ''))}\n\n### Assistant:\n{example.get('output', example.get('response', ''))}"}
            return dataset.map(format_long, remove_columns=dataset.column_names)

        elif "ultrafeedback" in name:
            # UltraFeedback often has 'instruction' and 'response' or 'text'
            if "text" not in dataset.column_names:
                def format_ultra(example):
                    return {"text": f"### User:\n{example['instruction']}\n\n### Assistant:\n{example['response']}"}
                return dataset.map(format_ultra, remove_columns=dataset.column_names)
        
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

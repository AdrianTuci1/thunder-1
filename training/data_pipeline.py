import os
import random
from typing import Iterable, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, interleave_datasets

from core.config_manager import THUNDER_CONFIG


class PackedTokenDataset(Dataset):
    def __init__(self, blocks: List[List[int]]):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        return {"input_ids": self.blocks[index]}


class StreamingBlockDataset(IterableDataset):
    """
    Real-time block packer for streamed datasets.
    Yields fixed-length token blocks as they are processed.
    """

    def __init__(self, tokenizer, dataset, block_size, max_blocks=None, eos_token_id=None, extra_params=None):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.eos_token_id = eos_token_id
        self.extra_params = extra_params or {}
        
        # [FIX] Lie to the DataLoader/HF to prevent automatic worker reduction
        self.num_shards = 1000 
        
        # We need access to formatting logic
        self.pipeline_helper = ThunderDataPipeline(tokenizer)

    def __iter__(self):
        import torch
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        token_buffer: List[int] = []
        blocks_yielded = 0
        
        # Patch get_worker_info so HuggingFace 'datasets' doesn't kill our workers
        import unittest.mock
        with unittest.mock.patch('torch.utils.data.get_worker_info', return_value=None):
            dataset_iterator = iter(self.dataset)

            for i, example in enumerate(dataset_iterator):
                if i % num_workers != worker_id:
                    continue
                # Determine which spec to use for formatting
                # interleave_datasets adds a 'label' or we can try to guess
                # but for now we use a generic extractor or check for fields
                text = self.pipeline_helper._extract_text_generic(example)
                
                if not text:
                    continue
                    
                token_ids = self.pipeline_helper._tokenize_text(text)
                if not token_ids:
                    continue

                token_buffer.extend(token_ids)
                
                # Add EOS between documents if configured
                if self.extra_params.get("eos_between_documents", True) and self.eos_token_id is not None:
                    token_buffer.append(self.eos_token_id)

                while len(token_buffer) >= self.block_size:
                    yield {"input_ids": torch.tensor(token_buffer[: self.block_size], dtype=torch.long)}
                    token_buffer = token_buffer[self.block_size :]
                    blocks_yielded += 1
                    
                    if self.max_blocks is not None and blocks_yielded >= self.max_blocks:
                        return


class ThunderDataPipeline:
    """
    Hugging Face dataset loader + constant-length block packer.
    This path is designed for a from-scratch dLLM with 2048-token windows.
    """

    def __init__(self, tokenizer, max_seq_length=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length or THUNDER_CONFIG["engine"]["max_seq_len"]
        self.block_size = THUNDER_CONFIG["pipeline"].get("block_size", self.max_seq_length)
        self.shuffle_seed = THUNDER_CONFIG["pipeline"].get("shuffle_seed", THUNDER_CONFIG["training"]["seed"])
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    def prepare_dataset(
        self,
        dataset_specs=None,
        augment=True,
        max_blocks: Optional[int] = None,
        max_documents_per_dataset: Optional[int] = None,
    ):
        dataset_specs = dataset_specs or THUNDER_CONFIG["pipeline"]["pretrain_hf_datasets"]
        max_blocks = max_blocks or THUNDER_CONFIG["training"].get("max_train_blocks")
        
        print(f"⚡ Thunder: Initializing real-time stream from {len(dataset_specs)} sources...")
        
        loaded_datasets = []
        weights = []
        
        for spec in dataset_specs:
            dataset_name = spec["path"]
            dataset_config = spec.get("name")
            split = spec.get("split", "train")
            
            # [FIX] Load as stream
            if os.path.exists(dataset_name):
                builder = "json" if dataset_name.endswith((".json", ".jsonl")) else "text"
                ds = load_dataset(builder, data_files=dataset_name, split=split, streaming=True)
            else:
                ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
            
            # [NEW] Handle sharding for Multi-GPU/Distributed training
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                # Interleave-aware sharding: each process takes a shard of the stream
                ds = ds.shard(num_shards=world_size, index=rank)
            
            # [FIX] Unify features strictly to avoid alignment errors between different metadata schemas.
            # For IterableDatasets, column_names can be None, so we use a robust mapping.
            text_field = spec.get("text_field", "text")
            ds = ds.map(lambda x: {"text": x.get(text_field, "")})
            
            # Filter to keep ONLY the 'text' column for interleaving
            try:
                ds = ds.select_columns(["text"])
            except Exception:
                # Fallback if select_columns fails on streaming
                pass
            
            loaded_datasets.append(ds)
            weights.append(spec.get("weight", 1.0))

        # Interleave sources according to weights
        if len(loaded_datasets) > 1:
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            dataset = interleave_datasets(loaded_datasets, probabilities=probs, seed=self.shuffle_seed, stopping_strategy="all_exhausted")
        else:
            dataset = loaded_datasets[0]

        return StreamingBlockDataset(
            tokenizer=self.tokenizer,
            dataset=dataset,
            block_size=self.block_size,
            max_blocks=max_blocks,
            eos_token_id=self.eos_token_id,
            extra_params=THUNDER_CONFIG["pipeline"]
        )

    def pack_texts(self, texts: Iterable[str], max_blocks: Optional[int] = None) -> List[List[int]]:
        blocks: List[List[int]] = []
        token_buffer: List[int] = []

        for text in texts:
            if not text:
                continue
            token_ids = self._tokenize_text(text)
            if not token_ids:
                continue

            token_buffer.extend(token_ids)
            if THUNDER_CONFIG["pipeline"].get("eos_between_documents", True) and self.eos_token_id is not None:
                token_buffer.append(self.eos_token_id)

            while len(token_buffer) >= self.block_size:
                blocks.append(token_buffer[: self.block_size])
                token_buffer = token_buffer[self.block_size :]
                if max_blocks is not None and len(blocks) >= max_blocks:
                    return blocks

        return blocks

    def _tokenize_text(self, text: str) -> List[int]:
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=100000,
            return_attention_mask=False,
        )
        return list(tokenized["input_ids"])

    def _collect_texts(self, dataset, spec, limit: int) -> List[str]:
        texts: List[str] = []
        
        # Use .take() only for streaming datasets to avoid IndexError on small local files
        if hasattr(dataset, "take") and spec.get("streaming", False):
            iterator = dataset.take(limit)
        else:
            iterator = dataset

        for example in iterator:
            normalized = self._extract_text(example, spec)
            if normalized:
                texts.append(normalized)
            if len(texts) >= limit:
                break

        return texts

    def _extract_text(self, example, spec) -> str:
        fmt = spec.get("format", "text")

        if fmt == "text":
            text_field = spec.get("text_field", "text")
            value = example.get(text_field, "")
            return value.strip() if isinstance(value, str) else ""

        if fmt in {"conversations", "messages"}:
            messages = example.get(spec.get("messages_field", "messages"), [])
            return self._format_messages(messages)

        if fmt == "prompt_response":
            prompt_field = spec.get("prompt_field", "prompt")
            response_field = spec.get("response_field", "response")
            prompt_value = example.get(prompt_field, "")
            response_value = example.get(response_field, "")

            if isinstance(response_value, list):
                response_value = self._format_messages(response_value)

            return self._join_turns(prompt_value, response_value)

        if fmt == "instruction_output":
            return self._join_turns(example.get("instruction", ""), example.get("output", ""))

        return ""

    def _extract_text_generic(self, example) -> str:
        """
        Attempts to extract text from a dictionary regardless of the specific spec format.
        Used for interleaved streaming where spec information might be lost or unified.
        """
        # 1. Check for standard text fields
        for field in ["text", "content", "body", "value"]:
            if field in example and isinstance(example[field], str):
                return example[field].strip()
        
        # 2. Check for conversation formats
        for field in ["conversations", "messages"]:
            if field in example and isinstance(example[field], list):
                return self._format_messages(example[field])
        
        # 3. Check for Prompt/Response
        if "prompt" in example and "response" in example:
            return self._join_turns(example["prompt"], example["response"])
            
        return ""

    def _format_messages(self, messages) -> str:
        rendered = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue

            if "role" in message:
                role = message["role"]
                content = message.get("content", "")
            else:
                role = "user" if message.get("from") == "human" else "assistant"
                content = message.get("value", "")

            if not content:
                continue

            label = "User" if role in {"user", "human"} else "Assistant"
            rendered.append(f"### {label}:\n{content.strip()}")

        return "\n\n".join(rendered).strip()

    def _join_turns(self, prompt, response) -> str:
        prompt = prompt.strip() if isinstance(prompt, str) else ""
        response = response.strip() if isinstance(response, str) else ""
        if not prompt and not response:
            return ""
        return f"### User:\n{prompt}\n\n### Assistant:\n{response}".strip()

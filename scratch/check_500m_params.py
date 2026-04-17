import os
import sys
import torch

sys.path.append(os.getcwd())

from core.config_manager import THUNDER_CONFIG
from core.scratch_dllm import ScratchDLMConfig, ThunderScratchDiffusionLM

def check_params():
    m_cfg = THUNDER_CONFIG["model"]
    cfg = ScratchDLMConfig(
        vocab_size=m_cfg["vocab_size"],
        embedding_dim=m_cfg["embedding_dim"],
        latent_dim=m_cfg["latent_dim"],
        ffn_hidden_size=m_cfg["ffn_hidden_size"],
        num_layers=m_cfg["num_layers"],
        num_attention_heads=m_cfg["num_attention_heads"],
        num_kv_heads=m_cfg["num_kv_heads"],
        max_seq_len=m_cfg["max_seq_len"],
        dropout=m_cfg["dropout"],
        self_conditioning=m_cfg["self_conditioning"],
        use_rope=m_cfg["use_rope"],
        rope_theta=m_cfg["rope_theta"],
    )
    
    model = ThunderScratchDiffusionLM(cfg)
    params = model.num_parameters()
    print(f"Total Parameters: {params:,}")
    
    # Estimate finish time for 10B tokens at estimated 30k tps
    tps = 30000
    total_tokens = 10_000_000_000
    seconds = total_tokens / tps
    hours = seconds / 3600
    print(f"Estimated time for 10B tokens at 30k tps: {hours:.2f} hours")

if __name__ == "__main__":
    check_params()

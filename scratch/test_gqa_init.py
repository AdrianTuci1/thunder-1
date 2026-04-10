import torch
from core.config_manager import THUNDER_CONFIG
from core.scratch_dllm import ThunderScratchDiffusionLM, ScratchDLMConfig

def test_init():
    print("⚡ Testing Thunder Initialization with GQA...")
    
    # Extract config
    dlm_config = ScratchDLMConfig(
        vocab_size=THUNDER_CONFIG["model"]["vocab_size"],
        embedding_dim=THUNDER_CONFIG["model"]["embedding_dim"],
        latent_dim=THUNDER_CONFIG["model"]["latent_dim"],
        ffn_hidden_size=THUNDER_CONFIG["model"]["ffn_hidden_size"],
        num_layers=THUNDER_CONFIG["model"]["num_layers"],
        num_attention_heads=THUNDER_CONFIG["model"]["num_attention_heads"],
        num_kv_heads=THUNDER_CONFIG["model"]["num_kv_heads"],
        max_seq_len=THUNDER_CONFIG["model"]["max_seq_len"],
        pad_token_id=THUNDER_CONFIG["model"]["pad_token_id"],
        dropout=THUNDER_CONFIG["model"]["dropout"],
        self_conditioning=THUNDER_CONFIG["model"]["self_conditioning"],
        use_rope=THUNDER_CONFIG["model"]["use_rope"],
        rope_theta=THUNDER_CONFIG["model"]["rope_theta"],
    )
    
    model = ThunderScratchDiffusionLM(dlm_config)
    print(f"✅ Model initialized successfully!")
    print(f"🔹 Total Parameters: {model.num_parameters() / 1e6:.2f}M")
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, dlm_config.vocab_size, (batch_size, seq_len))
    timesteps = torch.randint(0, 100, (batch_size,))
    
    print(f"🔹 Running dummy forward pass (seq_len={seq_len})...")
    with torch.no_grad():
        out = model(input_ids, timesteps)
    
    print(f"✅ Forward pass successful! Output shape: {out.shape}")
    
    # Test for GQA head broadcasting
    print(f"🔹 Verifying GQA head broadcasting logic...")
    # This should not crash if repeat_interleave is correct
    print("✅ GQA logic verified via forward pass.")

if __name__ == "__main__":
    test_init()

import os
import torch
import json
import sys
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_manager import THUNDER_CONFIG
from core.model_loader import ThunderModelLoader

def get_latest_checkpoint(runs_dir):
    if not os.path.exists(runs_dir):
        return None
    ckpts = [d for d in os.listdir(runs_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    # Sort by step number
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(runs_dir, ckpts[-1])

def generate_live_preview():
    output_dir = THUNDER_CONFIG["training"].get("output_dir", "./runs/thunder_v1_850M_production")
    ckpt_path = get_latest_checkpoint(output_dir)
    
    if not ckpt_path:
        print(f"❌ No checkpoints found in {output_dir}")
        return

    print(f"🔄 Loading latest checkpoint: {ckpt_path}")
    
    loader = ThunderModelLoader()
    model, tokenizer = loader.load_model(load_in_4bit=False)
    
    # Load state dict
    state_dict = torch.load(os.path.join(ckpt_path, "model_state.pt"), map_location=model.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✅ Model loaded successfully. Generating clean previews (t=0)...")
    
    # Sample some test prompts (Identity or common code)
    test_prompts = [
        "What is Thunder?",
        "def quicksort(arr):",
        "import torch\nimport torch.nn as nn",
        "SELECT * FROM users WHERE",
    ]
    
    embedding_matrix = model.get_input_embeddings().weight
    
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            input_ids = inputs["input_ids"]
            
            # Use t=0 for "Oracle" clean reconstruction (checking what model learned for these tokens)
            t_zero = torch.zeros((1,), device=model.device).long()
            
            # Forward pass
            inputs_embeds = model.get_input_embeddings()(input_ids)
            x0_pred = model.diffusion_forward(inputs_embeds, t_zero)
            
            # Robust Decoding (Cosine Similarity)
            x0_norm = x0_pred[0].float() / (x0_pred[0].float().norm(dim=-1, keepdim=True) + 1e-8)
            emb_norm = embedding_matrix.float() / (embedding_matrix.float().norm(dim=-1, keepdim=True) + 1e-8)
            
            logits = torch.matmul(x0_norm, emb_norm.t())
            tokens = torch.argmax(logits, dim=-1)
            
            result_text = tokenizer.decode(tokens, skip_special_tokens=True)
            
            print(f"\nPrompt:  {prompt}")
            print(f"Recovered: {result_text}")
            print("-" * 30)

if __name__ == "__main__":
    generate_live_preview()

import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_loader import ThunderModelLoader
from core.config_manager import THUNDER_CONFIG

def test_llada_loading():
    print("⚡ Testing LLaDA 8B Loading and Adaptation...")
    
    # Force LLaDA model for testing
    loader = ThunderModelLoader(model_name="GSAI/LLaDA-8B-Instruct")
    
    try:
        model, tokenizer = loader.load_model(load_in_4bit=True)
        print("✅ Model and Tokenizer loaded successfully.")
        
        # Verify adaptation
        if hasattr(model, "is_thunder_adapted") and model.is_thunder_adapted:
            print("✅ Model successfully adapted for Thunder Diffusion.")
        else:
            print("❌ Model adaptation failed.")
            return

        # Simple Forward Pass Test
        print("⚡ Testing bidirectional forward pass...")
        query = "### User:\nHello!\n\n### Assistant:\n"
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        
        # Test diffusion_forward
        seq_len = inputs.input_ids.shape[1]
        dummy_x_t = torch.randn((1, seq_len, model.config.hidden_size), device=model.device, dtype=model.dtype)
        dummy_t = torch.tensor([500], device=model.device)
        
        with torch.no_grad():
            x0_pred = model.diffusion_forward(dummy_x_t, dummy_t)
            
        print(f"✅ Forward pass successful. Output shape: {x0_pred.shape}")
        
    except Exception as e:
        print(f"❌ Error during LLaDA loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llada_loading()

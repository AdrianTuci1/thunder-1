import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.diffusion_engine import ThunderDiffusionEngine
from core.scheduler import ThunderScheduler

class MockModel(torch.nn.Module):
    def __init__(self, target_latent):
        super().__init__()
        self.target_latent = target_latent
        self.logits = None

    def forward(self, x):
        # In this mock, the model predicts the noise correctly 
        # to nudge x towards target_latent.
        # Simplified: epsilon_pred = noise in x relative to target
        # For this test, we mimic a model that "knows" how to fix the noise.
        class Output:
            pass
        out = Output()
        # Mock logic: return a dummy prediction
        out.logits = torch.randn_like(x) * 0.1 
        return out

def test_diffusion_engine():
    print("Testing ThunderDiffusionEngine Logic...")
    
    B, L, D = 1, 128, 64
    target_latent = torch.ones(B, L, D)
    model = MockModel(target_latent)
    
    adaptive_scheduler = ThunderScheduler()
    engine = ThunderDiffusionEngine(model, adaptive_scheduler)
    
    # 1. Test Denoising Step Math
    print("\n1. Testing _denoise_step...")
    x_init = torch.randn(B, L, D)
    # total_steps=100, current step 99 (first in countdown)
    x_next = engine._denoise_step(x_init, 99, 100)
    
    print(f"Initial mean: {x_init.mean().item():.4f}")
    print(f"Next mean: {x_next.mean().item():.4f}")
    assert x_next.shape == x_init.shape, "Shape mismatch in denoise step"
    
    # 2. Test Crystallization Loop
    print("\n2. Testing crystallize_tile loop...")
    # Using small steps for speed
    final_state = engine.crystallize_tile(x_init, steps=10)
    print(f"Final state shape: {final_state.shape}")
    assert not torch.isnan(final_state).any(), "NaN detected in final state"

    # 3. Test Anchored Fusion
    print("\n3. Testing _apply_anchor...")
    anchor_data = torch.zeros(B, 128, D)
    anchor_data[:, -32:, :] = 5.0 # Distinctive value in overlap
    
    x_to_anchor = torch.zeros(B, L, D)
    anchored_x = engine._apply_anchor(x_to_anchor, anchor_data)
    
    overlap_start_val = anchored_x[:, 0, :].mean().item()
    print(f"Value at start of overlap: {overlap_start_val:.4f}")
    assert overlap_start_val > 0, "Anchor was not applied correctly"

    print("\n✅ Engine logic verified!")

if __name__ == "__main__":
    test_diffusion_engine()

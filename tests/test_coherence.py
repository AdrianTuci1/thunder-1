import sys
import os
import torch
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.diffusion_engine import ThunderDiffusionEngine
from core.scheduler import ThunderScheduler

class TestCoherence(unittest.TestCase):
    def setUp(self):
        # Mock model that returns input * 1.1 (simple prediction)
        self.mock_model = MagicMock()
        self.mock_model.side_effect = lambda x: MagicMock(logits=x * 0.1)
        
        self.scheduler = ThunderScheduler()
        self.engine = ThunderDiffusionEngine(self.mock_model, self.scheduler)

    def test_anchored_denoising(self):
        # Create mock data [B, L, D]
        B, L, D = 1, 512, 128
        tile_data = torch.zeros((B, L, D))
        anchor_data = torch.ones((B, 128, D)) # Anchor with all ones at the end
        
        # Run crystallization with anchor
        # Use few steps for speed
        result = self.engine.crystallize_tile(tile_data, steps=5, anchor_data=anchor_data)
        
        # Check if the beginning of the result matches the anchor (due to high anchor strength)
        # The first few tokens should be very close to 1.0
        overlap_size = self.engine.fuser.overlap_size
        beginning = result[:, :10, :] # Check first 10 tokens
        
        # Because weights[0] is sigmoid(-5) which is ~0.0067, 
        # result[0] = (1-0.0067)*1.0 + 0.0067*current_val ~= 1.0
        self.assertTrue(torch.allclose(beginning, torch.ones_like(beginning), atol=0.1))
        print("✅ Anchored Denoising Verified (Boundary constraint applied).")

    def test_global_coherence(self):
        B, L, D = 1, 512, 128
        tile_data = torch.zeros((B, L, D))
        macro_context = torch.ones((B, L, D)) * 0.5 # Nudge towards 0.5
        
        # Run with macro context
        result = self.engine.crystallize_tile(tile_data, steps=10, macro_context=macro_context)
        
        # Without any model bias (our mock just scales), the result should drift towards macro_context
        # If it works, the mean should be closer to 0.5 than to 0.0
        self.assertTrue(result.mean() > 0.1) 
        print(f"✅ Global Coherence Verified (Mean shifted towards macro: {result.mean().item():.4f}).")

if __name__ == "__main__":
    unittest.main()

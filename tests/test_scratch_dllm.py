import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch

    from core.scratch_dllm import ScratchDLMConfig, ThunderScratchDiffusionLM, build_bidirectional_attention_mask
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is required for scratch model tests")
class TestScratchDLM(unittest.TestCase):
    def test_bidirectional_attention_mask_is_not_causal(self):
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        expanded = build_bidirectional_attention_mask(attention_mask)
        self.assertEqual(expanded.shape, (1, 1, 1, 4))
        self.assertTrue(bool(expanded[0, 0, 0, 2].item()))
        self.assertFalse(bool(expanded[0, 0, 0, 3].item()))

    def test_scratch_dllm_forward_shapes(self):
        config = ScratchDLMConfig(
            vocab_size=128,
            embedding_dim=64,
            latent_dim=64,
            ffn_hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            max_seq_len=32,
            pad_token_id=0,
        )
        model = ThunderScratchDiffusionLM(config, diffusion_steps=32)

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        timesteps = torch.randint(0, 32, (2,))

        outputs = model(input_ids=input_ids, timesteps=timesteps, attention_mask=attention_mask)
        self.assertEqual(outputs.shape, (2, 16, config.embedding_dim))
        self.assertGreater(model.num_parameters(), 0)


if __name__ == "__main__":
    unittest.main()

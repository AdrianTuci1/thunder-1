import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn as nn

    from core.diffusion_engine import ThunderDiffusionEngine
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


if HAS_TORCH:
    class DummyDiffusionModel(nn.Module):
        def __init__(self, vocab_size=32, hidden_size=16):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.config = type("Config", (), {"hidden_size": hidden_size})()

        @property
        def device(self):
            return self.embedding.weight.device

        @property
        def dtype(self):
            return self.embedding.weight.dtype

        def get_input_embeddings(self):
            return self.embedding

        def diffusion_forward(self, x_t, t, attention_mask=None, self_cond=None):
            if self_cond is not None:
                x_t = x_t + 0.1 * self_cond
            return x_t


@unittest.skipUnless(HAS_TORCH, "torch is required for coherence tests")
class TestCoherence(unittest.TestCase):
    def test_generation_keeps_prompt_anchor_shape_consistent(self):
        model = DummyDiffusionModel()
        engine = ThunderDiffusionEngine(model)

        prompt_ids = torch.tensor([[1, 2, 3]])
        prompt_embeds = model.get_input_embeddings()(prompt_ids)
        embedding_matrix = model.get_input_embeddings().weight.detach()

        _, final_tokens = engine.generate(
            shape=(1, 8, model.config.hidden_size),
            embedding_matrix=embedding_matrix,
            steps=4,
            prompt_embeds=prompt_embeds,
            anchor_len=prompt_ids.shape[1],
            guidance_scale=1.0,
            max_new_tokens=5,
        )

        self.assertEqual(final_tokens.shape[0], 1)
        self.assertEqual(final_tokens.shape[1], prompt_ids.shape[1] + 5)


if __name__ == "__main__":
    unittest.main()

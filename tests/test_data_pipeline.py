import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from training.data_pipeline import ThunderDataPipeline
    HAS_PIPELINE = True
except ModuleNotFoundError:
    HAS_PIPELINE = False


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, return_attention_mask=False):
        token_ids = [len(token) + 1 for token in text.split()]
        return {"input_ids": token_ids}


@unittest.skipUnless(HAS_PIPELINE, "torch-backed data pipeline is not available in this Python environment")
class TestDataPipeline(unittest.TestCase):
    def test_constant_length_packing_produces_fixed_blocks(self):
        pipeline = ThunderDataPipeline(FakeTokenizer(), max_seq_length=8)
        pipeline.block_size = 8

        blocks = pipeline.pack_texts(
            [
                "alpha beta gamma delta",
                "epsilon zeta eta theta iota",
                "kappa lambda mu nu xi omicron",
            ],
            max_blocks=2,
        )

        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(len(block) == 8 for block in blocks))


if __name__ == "__main__":
    unittest.main()

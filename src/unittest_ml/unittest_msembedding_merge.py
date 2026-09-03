from __future__ import annotations

import os
import unittest

import torch

from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from flex.ml_algorithms.lora.impl.lora_ms import MSEmbedding


class TestMSEmbeddingMerge(unittest.TestCase):
    def test_eval_merge_matches_embedding_weight_shape(self):
        emb = MSEmbedding(
            num_embeddings=10,
            embedding_dim=4,
            r=2,
            lora_alpha=2,
            merge_weights=True,
        )
        with torch.no_grad():
            emb.lora_A.normal_()
            emb.lora_B.normal_()

        original = emb.weight.detach().clone()
        delta = (emb.lora_B @ emb.lora_A).T * emb.scaling

        self.assertEqual(tuple(delta.shape), tuple(emb.weight.shape))

        emb.eval()
        torch.testing.assert_close(emb.weight, original + delta)
        self.assertTrue(emb.merged)

        emb.train()
        torch.testing.assert_close(emb.weight, original)
        self.assertFalse(emb.merged)

        emb.eval()
        torch.testing.assert_close(emb.weight, original + delta)
        self.assertTrue(emb.merged)


if __name__ == "__main__":
    unittest.main()

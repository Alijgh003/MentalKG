import unittest

import numpy as np

from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder


class FakeEmbedder(OpenAICompatibleEmbedder):
    def _post(self, payload):
        dimension = payload["dimensions"]
        return {
            "data": [
                {"index": index, "embedding": [float(index + 1)] * dimension}
                for index, _ in reversed(list(enumerate(payload["input"])))
            ]
        }


class EmbeddingTests(unittest.TestCase):
    def test_array_input_returns_float32_matrix_in_input_order(self):
        embedder = FakeEmbedder(
            EmbeddingConfig(base_url="http://unused/v1", model="test", batch_size=2)
        )
        result = embedder.embed(["a", "b", "c"], dimension=3)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0], [1, 1, 1])
        np.testing.assert_array_equal(result[1], [2, 2, 2])

    def test_empty_input(self):
        result = FakeEmbedder(
            EmbeddingConfig(base_url="http://unused/v1", model="test")
        ).embed([], dimension=3)
        self.assertEqual(result.shape, (0, 3))

    def test_rejects_wrong_dimension(self):
        class WrongDimension(FakeEmbedder):
            def _post(self, payload):
                return {"data": [{"index": 0, "embedding": [1.0]}]}

        with self.assertRaisesRegex(ValueError, "Expected embedding shape"):
            WrongDimension(
                EmbeddingConfig(base_url="http://unused/v1", model="test")
            ).embed(["a"], dimension=3)

    def test_dimension_is_selected_per_call(self):
        embedder = FakeEmbedder(EmbeddingConfig(base_url="http://unused/v1", model="test"))
        self.assertEqual(embedder.embed(["a"], dimension=128).shape, (1, 128))
        self.assertEqual(embedder.embed(["a"], dimension=512).shape, (1, 512))


if __name__ == "__main__":
    unittest.main()

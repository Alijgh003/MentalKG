import unittest

import numpy as np

from kg_pipeline.clustering import ClusterPolicy, choose_cluster_count, cluster_vectors


class ClusteringTests(unittest.TestCase):
    def test_cluster_count_is_bounded(self):
        policy = ClusterPolicy(sqrt_factor=3.0)
        self.assertEqual(choose_cluster_count(0, policy), 0)
        self.assertEqual(choose_cluster_count(1, policy), 1)
        self.assertEqual(choose_cluster_count(4, policy), 4)
        self.assertEqual(choose_cluster_count(100, policy), 30)

    def test_single_cluster_returns_cosine_scores(self):
        matrix = np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
        labels, centroids, similarities = cluster_vectors(matrix, 1)
        np.testing.assert_array_equal(labels, [0, 0])
        self.assertEqual(centroids.shape, (1, 2))
        self.assertTrue(np.all(similarities <= 1.000001))


if __name__ == "__main__":
    unittest.main()

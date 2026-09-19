import unittest

import numpy as np

from kg_pipeline.consolidation import UniqueTerm, consolidate_cluster, normalize_text


def term(text, vector, mentions=1):
    return UniqueTerm(text, normalize_text(text), text, mentions, "test:0", np.asarray(vector))


class ConsolidationTests(unittest.TestCase):
    def test_normalization_matches_embedding_ingestion(self):
        self.assertEqual(normalize_text("  Depressed   Mood "), "depressed   mood")

    def test_requires_direct_similarity_to_representative(self):
        terms = [
            term("common", [1.0, 0.0], mentions=10),
            term("near", [0.9, 0.1]),
            term("far", [0.0, 1.0]),
        ]
        groups = consolidate_cluster(terms, 0.87)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].representative.text, "common")
        self.assertEqual({member.text for member in groups[0].members}, {"common", "near"})
        self.assertEqual(groups[1].members[0].text, "far")

    def test_threshold_is_strict(self):
        terms = [term("a", [1.0, 0.0]), term("b", [0.87, np.sqrt(1 - 0.87**2)])]
        groups = consolidate_cluster(terms, 0.87)
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()

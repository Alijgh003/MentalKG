import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from methods.datasets import load_samples
from methods.hipporag.ppr import run_personalized_pagerank, transition_matrix
from methods.hipporag.recognition import _clean_triples, render_fact_query
from methods.hipporag.method import HippoRAG2Config, HippoRAG2Method
from methods.registry import available_methods
from methods.telemetry import StageRecorder
from methods.vanilla_rag.method import VanillaRAGConfig
from kg_pipeline.reranking import OpenAICompatibleReranker, RerankerConfig
from scripts.run_method import stage_output_for_display


class PPRTests(unittest.TestCase):
    def test_personalization_propagates_over_graph(self):
        adjacency = sparse.csr_matrix(np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]))
        transition, dangling = transition_matrix(adjacency)
        scores, iterations = run_personalized_pagerank(
            transition,
            np.array([1.0, 0.0, 0.0]),
            damping=0.5,
            dangling=dangling,
        )
        self.assertAlmostEqual(float(scores.sum()), 1.0)
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(iterations, 0)

    def test_negative_and_nan_reset_weights_are_ignored(self):
        transition, dangling = transition_matrix(sparse.eye(2, format="csr"))
        scores, _ = run_personalized_pagerank(
            transition,
            np.array([np.nan, 2.0]),
            dangling=dangling,
        )
        np.testing.assert_allclose(scores, [0.0, 1.0])


class DatasetTests(unittest.TestCase):
    def test_load_samples_supports_limit_offset_and_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "Demo"
            dataset_dir.mkdir()
            with (dataset_dir / "test.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["benchmark_id", "gold_label", "post"])
                writer.writeheader()
                writer.writerow({"benchmark_id": "one", "gold_label": "a", "post": "first"})
                writer.writerow({"benchmark_id": "two", "gold_label": "b", "post": "second"})
            samples = load_samples("demo", "test", limit=1, offset=1, root=root)
        self.assertEqual(samples[0].sample_id, "two")
        self.assertEqual(samples[0].text, "second")
        self.assertEqual(samples[0].dataset_row_index, 1)


class TelemetryTests(unittest.TestCase):
    def test_stage_records_latency_counts_and_error(self):
        recorder = StageRecorder(sample_id="sample-1")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with recorder.stage("llm_stage") as event:
                event["llm_calls"] = 1
                event["input_tokens"] = 12
                raise RuntimeError("boom")
        recorded = recorder.events[0]
        self.assertEqual(recorded["status"], "error")
        self.assertEqual(recorded["llm_calls"], 1)
        self.assertEqual(recorded["input_tokens"], 12)
        self.assertGreaterEqual(recorded["latency_ms"], 0)

    def test_passage_retrieval_preview_keeps_only_first_ten(self):
        output, metadata = stage_output_for_display("passage_retrieval", list(range(200)))
        self.assertEqual(output, list(range(10)))
        self.assertEqual(metadata, {"showing": 10, "total": 200, "truncated": True})


class VanillaRAGTests(unittest.TestCase):
    def test_method_is_registered_and_defaults_are_bounded(self):
        self.assertIn("vanilla_rag", available_methods())
        config = VanillaRAGConfig()
        self.assertEqual(config.retrieval_top_k, 50)
        self.assertEqual(config.rerank_top_k, 10)
        self.assertEqual(config.qa_top_k, 5)

    def test_reranker_parses_and_validates_ranked_results(self):
        reranker = OpenAICompatibleReranker(RerankerConfig(
            base_url="http://reranker.invalid/v1",
            api_key="test",
            model="test-reranker",
        ))
        reranker._post = lambda payload: {"results": [
            {"index": 1, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.42},
        ]}
        hits = reranker.rerank("query", ["first", "second"], top_n=2)
        self.assertEqual([hit.index for hit in hits], [1, 0])
        self.assertEqual([hit.score for hit in hits], [0.91, 0.42])


class FactRecognitionTests(unittest.TestCase):
    def test_generated_fact_queries_are_cleaned_deduplicated_and_limited(self):
        triples = _clean_triples([
            {"subject": " sleep ", "predicate": "is disrupted by", "object": "anxiety"},
            {"subject": "sleep", "predicate": "is disrupted by", "object": "anxiety"},
            {"subject": "", "predicate": "has", "object": "missing subject"},
            {"subject": "mood", "predicate": "remains", "object": "low"},
        ], limit=2)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0]["subject"], "sleep")
        self.assertEqual(render_fact_query(triples[0]), "sleep is disrupted by anxiety")

    def test_fact_retrieval_defaults_use_broad_pool_then_final_top_k(self):
        config = HippoRAG2Config()
        self.assertEqual(config.fact_retrieval_top_k_per_query, 15)
        self.assertEqual(config.fact_filter_candidate_limit, 20)
        self.assertEqual(config.final_fact_top_k, 10)
        self.assertEqual(config.passage_similarity_threshold, 0.50)
        self.assertEqual(config.passage_seed_mass_ratio, 0.10)

    def test_global_fact_shortlist_preserves_best_candidate_for_each_query(self):
        candidates = [
            {"id": "q1-best", "raw_score": 0.95, "matched_queries": [{"query_rank": 1}]},
            {"id": "q1-second", "raw_score": 0.90, "matched_queries": [{"query_rank": 1}]},
            {"id": "q2-best", "raw_score": 0.70, "matched_queries": [{"query_rank": 2}]},
        ]
        shortlisted = HippoRAG2Method._fact_filter_shortlist(
            candidates, query_count=2, limit=2
        )
        self.assertEqual([row["id"] for row in shortlisted], ["q1-best", "q2-best"])


if __name__ == "__main__":
    unittest.main()

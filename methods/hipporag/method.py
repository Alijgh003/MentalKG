from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
import psycopg
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.env import PROJECT_ENV_FILE
from config.settings import DatabaseSettings
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig
from methods.base import BenchmarkSample, MethodResult
from methods.telemetry import StageRecorder

from .graph import GraphProjection, load_graph_projection
from .ppr import run_personalized_pagerank
from .recognition import FactRecognizer, FactRelevanceFilter, render_fact_query
from .answering import EvidenceAnswerer


logger = logging.getLogger(__name__)
SEED_TYPES = frozenset({"symptom", "behavior", "disorder", "concept"})
STOP_STAGES = (
    "fact_generation",
    "fact_retrieval",
    "passage_retrieval",
    "seed_weighting",
    "ppr",
    "passage_ranking",
    "answer_generation",
)


class HippoRAG2Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HIPPORAG2_",
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    fact_retrieval_top_k_per_query: int = Field(default=15, gt=0)
    fact_filter_candidate_limit: int = Field(default=20, gt=0)
    final_fact_top_k: int = Field(default=10, gt=0)
    linking_top_k: int | None = Field(default=None, gt=0)
    fact_query_limit: int = Field(default=5, gt=0)
    passage_seed_top_k: int = Field(default=200, gt=0)
    retrieval_top_k: int = Field(default=20, gt=0)
    qa_top_k: int = Field(default=5, gt=0)
    passage_similarity_threshold: float = Field(default=0.50, ge=-1, le=1)
    passage_seed_mass_ratio: float = Field(default=0.10, ge=0)
    damping: float = Field(default=0.5, gt=0, lt=1)
    ppr_tolerance: float = Field(default=1e-10, gt=0)
    ppr_max_iterations: int = Field(default=200, gt=0)
    recognition_mode: str = "llm"


class HippoRAG2Method:
    name = "hipporag2"

    def __init__(
        self, *, generate_answer: bool = False, config_overrides: dict | None = None,
        stop_after: str | None = None,
    ):
        startup = StageRecorder(scope="startup", method=self.name)
        self.config = HippoRAG2Config(**(config_overrides or {}))
        if stop_after is not None and stop_after not in STOP_STAGES:
            raise ValueError(f"Unknown stop stage {stop_after!r}; choose from {', '.join(STOP_STAGES)}")
        self.stop_after = stop_after
        self.generate_answer = generate_answer or stop_after == "answer_generation"
        with startup.stage("recognition_program_init", mode=self.config.recognition_mode):
            self.recognizer = FactRecognizer(
                self.config.recognition_mode,
                max_queries=self.config.fact_query_limit,
            )
        self.fact_filter = None
        if self.config.recognition_mode == "llm" and stop_after != "fact_generation":
            with startup.stage("fact_filter_program_init"):
                self.fact_filter = FactRelevanceFilter()
        self.answerer = None
        if self.generate_answer:
            with startup.stage("answer_program_init"):
                self.answerer = EvidenceAnswerer()
        self.connection = None
        self.embedder = None
        self.store = None
        self.graph = None
        # Fact generation is deliberately independently testable: it needs no
        # database, Milvus, embedding service, or graph snapshot.
        if stop_after != "fact_generation":
            with startup.stage("postgres_connect"):
                source_dsn = DatabaseSettings().database_url
                self.connection = psycopg.connect(target_dsn(source_dsn, database_name_from_env()))
            with startup.stage("embedding_client_init"):
                self.embedder = OpenAICompatibleEmbedder(EmbeddingConfig.from_env())
            with startup.stage("milvus_connect"):
                self.store = KnowledgeGraphVectorStore(MilvusConfig.from_env())
            if stop_after not in {"fact_retrieval", "passage_retrieval"}:
                with startup.stage("graph_projection_load") as event:
                    self.graph = load_graph_projection(self.connection)
                    event["details"].update({
                        "nodes": len(self.graph.node_ids),
                        "entities": len(self.graph.entity_types),
                        "chunks": len(self.graph.chunk_indices),
                        "transition_nonzeros": int(self.graph.transition.nnz),
                        "dangling_nodes": int(self.graph.dangling.sum()),
                    })
        self.startup_stages = startup.events

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()

    def _search(
        self, collection: str, query: str, dimension: int, limit: int,
        output_fields, recorder: StageRecorder, prefix: str,
    ):
        with recorder.stage(f"{prefix}_query_embedding", dimension=dimension) as event:
            event["embedding_calls"] = 1
            vector = self.embedder.embed([query], dimension=dimension)[0].tolist()
        with recorder.stage(f"{prefix}_milvus_search", collection=collection, limit=limit) as event:
            event["retrieval_calls"] = 1
            hits = self.store.search(
                collection,
                [vector],
                limit=limit,
                output_fields=output_fields,
            )[0]
            event["details"]["hits"] = len(hits)
            return hits

    @staticmethod
    def _normalize_hit_scores(rows: list[dict]) -> list[dict]:
        if not rows:
            return rows
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        score_range = float(scores.max() - scores.min())
        normalized = np.ones_like(scores) if score_range == 0 else (scores - scores.min()) / score_range
        for row, score in zip(rows, normalized, strict=True):
            row["score"] = float(score)
        return rows

    def _fact_candidates(
        self, queries: list[dict[str, str]], question: str, recorder: StageRecorder,
    ) -> list[dict]:
        """Fuse all query hits, globally shortlist them, then make one LLM filter call."""
        merged: dict[str, dict] = {}
        self._last_filter_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "queries": []}
        per_query_k = self.config.linking_top_k or self.config.fact_retrieval_top_k_per_query
        total_raw_hits = 0
        with recorder.stage(
            "fact_retrieval",
            queries=len(queries),
            per_query_k=per_query_k,
            filter_candidate_limit=self.config.fact_filter_candidate_limit,
            final_k=self.config.final_fact_top_k,
        ) as event:
            for query_rank, triple in enumerate(queries, start=1):
                query = render_fact_query(triple)
                hits = self._search(
                    self.store.config.triple_collection,
                    query,
                    self.store.config.triple_dimension,
                    per_query_k,
                    ("text", "subject_text", "predicate", "object_text"),
                    recorder,
                    f"fact_query_{query_rank}",
                )
                total_raw_hits += len(hits)
                for hit_rank, hit in enumerate(hits, start=1):
                    entity = hit.get("entity", {})
                    fact_id = str(hit["id"])
                    raw_score = float(hit.get("distance", 0.0))
                    match = {
                        "query_rank": query_rank,
                        "query": triple,
                        "query_text": query,
                        "hit_rank": hit_rank,
                        "score": raw_score,
                    }
                    existing = merged.get(fact_id)
                    if existing is None:
                        merged[fact_id] = {
                            "id": fact_id,
                            "score": raw_score,
                            "raw_score": raw_score,
                            "text": str(entity.get("text") or ""),
                            "subject_text": str(entity.get("subject_text") or ""),
                            "predicate": str(entity.get("predicate") or ""),
                            "object_text": str(entity.get("object_text") or ""),
                            "matched_queries": [match],
                        }
                    else:
                        existing["raw_score"] = max(existing["raw_score"], raw_score)
                        existing["score"] = existing["raw_score"]
                        existing["matched_queries"].append(match)
            broad_candidates = sorted(
                merged.values(), key=lambda row: row["raw_score"], reverse=True
            )
            candidates = self._fact_filter_shortlist(
                broad_candidates,
                query_count=len(queries),
                limit=self.config.fact_filter_candidate_limit,
            )
            if self.fact_filter is not None and candidates:
                with recorder.stage("fact_relevance_filter", candidates=len(candidates)) as filter_event:
                    filter_event["llm_calls"] = 1
                    filtered = self.fact_filter.filter(
                        question,
                        queries,
                        candidates,
                        max_selected=self.config.final_fact_top_k,
                    )
                    filter_event["input_tokens"] = filtered.input_tokens
                    filter_event["output_tokens"] = filtered.output_tokens
                    filter_event["details"].update({
                        "selected": len(filtered.selected_ids),
                        "selected_ids": list(filtered.selected_ids),
                        "errors": list(filtered.errors),
                    })
                self._last_filter_usage.update({
                    "calls": 1,
                    "input_tokens": filtered.input_tokens,
                    "output_tokens": filtered.output_tokens,
                    "queries": [{
                        "mode": "global",
                        "query_count": len(queries),
                        "candidate_count": len(candidates),
                        "selected_ids": list(filtered.selected_ids),
                        "errors": list(filtered.errors),
                    }],
                })
                selected_ids = set(filtered.selected_ids)
                candidates = [row for row in candidates if row["id"] in selected_ids]
            candidates.sort(key=lambda row: row["raw_score"], reverse=True)
            # Normalize only the globally filtered pool, then enforce the final
            # graph-seed limit even if a provider returned too many IDs.
            self._normalize_hit_scores(candidates)
            retained = candidates[: self.config.final_fact_top_k]
            event["details"].update({
                "unique_hits_before_filter": len(broad_candidates),
                "presented_to_filter": min(
                    len(broad_candidates), self.config.fact_filter_candidate_limit
                ),
                "retained_hits": len(retained),
                "total_hits_before_deduplication": total_raw_hits,
                "normalization_raw_min": min((row["raw_score"] for row in candidates), default=None),
                "normalization_raw_max": max((row["raw_score"] for row in candidates), default=None),
            })
            return retained

    @staticmethod
    def _fact_filter_shortlist(candidates: list[dict], *, query_count: int, limit: int) -> list[dict]:
        """Keep each query's best hit when possible, then fill by global cosine score."""
        selected: list[dict] = []
        selected_ids: set[str] = set()
        for query_rank in range(1, query_count + 1):
            candidate = next((
                row for row in candidates
                if any(match["query_rank"] == query_rank for match in row["matched_queries"])
            ), None)
            if candidate is not None and candidate["id"] not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate["id"])
            if len(selected) >= limit:
                return selected
        for candidate in candidates:
            if candidate["id"] not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate["id"])
            if len(selected) >= limit:
                break
        return selected

    def _passage_candidates(self, query: str, recorder: StageRecorder) -> list[dict]:
        hits = self._search(
            self.store.config.source_chunk_collection,
            query,
            self.store.config.source_chunk_dimension,
            self.config.passage_seed_top_k,
            ("text",),
            recorder,
            "passage",
        )
        rows = [
            {
                "id": str(hit["id"]),
                "score": float(hit.get("distance", 0.0)),
                "raw_score": float(hit.get("distance", 0.0)),
                "text": str(hit.get("entity", {}).get("text") or ""),
            }
            for hit in hits
        ]
        return self._normalize_hit_scores(rows)

    def _load_fact_endpoints(self, fact_ids: list[str]) -> dict[str, tuple[str, str, str]]:
        if not fact_ids:
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT id::text,subject_id::text,predicate,object_id::text "
                "FROM facts WHERE id = ANY(%s::uuid[])",
                (fact_ids,),
            )
            return {row[0]: (row[1], row[2], row[3]) for row in cursor}

    def run(self, sample: BenchmarkSample) -> MethodResult:
        started = time.perf_counter()
        recorder = StageRecorder(
            method=self.name,
            dataset=sample.dataset,
            split=sample.split,
            dataset_row_index=sample.dataset_row_index,
            source_row_index=sample.source_row_index,
            sample_id=sample.sample_id,
        )
        self.last_stages = recorder.events
        stage_outputs: dict[str, object] = {}
        self._last_filter_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "queries": []}

        with recorder.stage("fact_generation", mode=self.config.recognition_mode) as event:
            event["llm_calls"] = 1 if self.config.recognition_mode == "llm" else 0
            generation = self.recognizer.generate(sample.text)
            event["input_tokens"] = generation.input_tokens
            event["output_tokens"] = generation.output_tokens
            event["details"]["generated_triples"] = len(generation.triples)
        generated_queries = [
            {"query_id": f"fq{rank}", **triple, "text": render_fact_query(triple)}
            for rank, triple in enumerate(generation.triples, start=1)
        ]
        stage_outputs["fact_generation"] = generated_queries
        recognition_metadata = {
            "mode": self.config.recognition_mode,
            "generated_fact_queries": generated_queries,
            "raw_prediction": generation.raw_response,
        }
        if self.stop_after == "fact_generation":
            return self._result(
                recorder, stage_outputs, generation,
                stopping_reason="stopped_after_fact_generation",
                method_metadata={"recognition": recognition_metadata},
            )

        fact_candidates = self._fact_candidates(generation.triples, sample.text, recorder)
        stage_outputs["fact_filtering"] = self._last_filter_usage["queries"]
        with recorder.stage("fact_endpoint_lookup", facts=len(fact_candidates)):
            endpoints = self._load_fact_endpoints([fact["id"] for fact in fact_candidates])
        retrieved_facts = self._fact_trace(fact_candidates, endpoints)
        stage_outputs["fact_retrieval"] = retrieved_facts
        recognition_metadata["retrieved_fact_ids"] = [fact["id"] for fact in fact_candidates]
        if self.stop_after == "fact_retrieval":
            return self._result(
                recorder, stage_outputs, generation,
                triples=retrieved_facts,
                stopping_reason="stopped_after_fact_retrieval",
                method_metadata={"recognition": recognition_metadata},
            )

        passage_candidates = self._passage_candidates(sample.text, recorder)
        passage_candidates = [
            row for row in passage_candidates
            if row["raw_score"] >= self.config.passage_similarity_threshold
        ]
        dense_passages = self._passage_trace(passage_candidates)
        stage_outputs["passage_retrieval"] = dense_passages
        if self.stop_after == "passage_retrieval":
            return self._result(
                recorder, stage_outputs, generation,
                triples=retrieved_facts,
                passages=dense_passages,
                stopping_reason="stopped_after_passage_retrieval",
                method_metadata={"recognition": recognition_metadata},
            )

        with recorder.stage("entity_seed_weighting", selected_facts=len(fact_candidates)) as event:
            reset_weights = np.zeros(len(self.graph.node_ids), dtype=np.float64)
            seed_scores: dict[str, list[float]] = defaultdict(list)
            for fact in fact_candidates:
                endpoint = endpoints.get(fact["id"])
                if endpoint is None:
                    continue
                subject_id, _, object_id = endpoint
                for entity_id in (subject_id, object_id):
                    if self.graph.entity_types.get(entity_id) in SEED_TYPES:
                        degree_penalty = max(self.graph.entity_chunk_counts.get(entity_id, 1), 1)
                        seed_scores[entity_id].append(max(fact["score"], 0.0) / degree_penalty)
            for entity_id, scores in seed_scores.items():
                reset_weights[self.graph.node_index[f"entity:{entity_id}"]] = float(np.mean(scores))
            event["details"]["seed_entities"] = len(seed_scores)
        seed_trace = [
            {
                "entity_id": entity_id,
                "type": self.graph.entity_types.get(entity_id),
                "score": float(np.mean(values)),
                "rank": rank,
                "source_query_ids": ["q1"],
            }
            for rank, (entity_id, values) in enumerate(
                sorted(seed_scores.items(), key=lambda item: np.mean(item[1]), reverse=True), start=1
            )
        ]
        stage_outputs["seed_weighting"] = seed_trace
        if self.stop_after == "seed_weighting":
            return self._result(
                recorder, stage_outputs, generation,
                seed_entities=seed_trace,
                triples=retrieved_facts,
                passages=dense_passages,
                stopping_reason="stopped_after_seed_weighting",
                method_metadata={"recognition": recognition_metadata},
            )

        # Upstream HippoRAG 2 falls back to dense passage retrieval when
        # recognition memory returns no fact. Our typed policy also requires at
        # least one seed-eligible endpoint before graph propagation.
        if not fact_candidates or not seed_scores:
            stage_outputs["ppr"] = {
                "skipped": True,
                "reason": "no_retrieved_facts" if not fact_candidates else "no_seed_eligible_endpoints",
            }
            if self.stop_after == "ppr":
                return self._result(
                    recorder, stage_outputs, generation,
                    seed_entities=seed_trace,
                    triples=retrieved_facts,
                    passages=dense_passages,
                    stopping_reason="ppr_skipped",
                    method_metadata={"recognition": recognition_metadata},
                )
            with recorder.stage("passage_hydration", mode="dense_fallback"):
                ranked = self._hydrate_passages(
                    sorted(passage_candidates, key=lambda item: item["score"], reverse=True)[
                        : self.config.retrieval_top_k
                    ]
                )
            ranked_trace = self._passage_trace(ranked)
            stage_outputs["passage_ranking"] = ranked_trace
            result = self._result(
                recorder, stage_outputs, generation,
                passages=ranked_trace,
                triples=retrieved_facts,
                stopping_reason="dense_fallback",
                method_metadata={"recognition": recognition_metadata},
            )
            return self._maybe_answer(sample, result, recorder)

        with recorder.stage("passage_seed_weighting", candidates=len(passage_candidates)) as event:
            applied = 0
            passage_weights: list[tuple[int, float]] = []
            for passage in passage_candidates:
                node_index = self.graph.node_index.get(f"chunk:{passage['id']}")
                if node_index is not None:
                    passage_weights.append((node_index, max(passage["score"], 0.0)))
                    applied += 1
            entity_mass = float(reset_weights.sum())
            raw_passage_mass = sum(weight for _, weight in passage_weights)
            passage_budget = entity_mass * self.config.passage_seed_mass_ratio
            scale = passage_budget / raw_passage_mass if raw_passage_mass > 0 else 0.0
            for node_index, weight in passage_weights:
                reset_weights[node_index] = weight * scale
            event["details"]["applied"] = applied
            event["details"].update({
                "similarity_threshold": self.config.passage_similarity_threshold,
                "entity_mass": entity_mass,
                "passage_mass": passage_budget if raw_passage_mass > 0 else 0.0,
                "passage_seed_mass_ratio": self.config.passage_seed_mass_ratio,
            })

        if reset_weights.sum() <= 0:
            stage_outputs["ppr"] = {"skipped": True, "reason": "zero_reset_weight"}
            if self.stop_after == "ppr":
                return self._result(
                    recorder, stage_outputs, generation,
                    seed_entities=seed_trace,
                    triples=retrieved_facts,
                    passages=dense_passages,
                    stopping_reason="ppr_skipped",
                    method_metadata={"recognition": recognition_metadata},
                )
            with recorder.stage("passage_hydration", mode="zero_reset_fallback"):
                ranked = self._hydrate_passages(
                    sorted(passage_candidates, key=lambda item: item["score"], reverse=True)[
                        : self.config.retrieval_top_k
                    ]
                )
            ranked_trace = self._passage_trace(ranked)
            stage_outputs["passage_ranking"] = ranked_trace
            result = self._result(
                recorder, stage_outputs, generation,
                passages=ranked_trace,
                triples=retrieved_facts,
                stopping_reason="dense_fallback",
                method_metadata={"recognition": recognition_metadata},
            )
            return self._maybe_answer(sample, result, recorder)

        with recorder.stage("personalized_pagerank", damping=self.config.damping) as event:
            scores, iterations = run_personalized_pagerank(
                self.graph.transition,
                reset_weights,
                damping=self.config.damping,
                dangling=self.graph.dangling,
                tolerance=self.config.ppr_tolerance,
                max_iterations=self.config.ppr_max_iterations,
            )
            event["details"]["iterations"] = iterations
        stage_outputs["ppr"] = {
            "iterations": iterations,
            "top_nodes": [
                {"node_id": self.graph.node_ids[index], "score": float(scores[index])}
                for index in np.argsort(scores)[::-1][: self.config.retrieval_top_k]
            ],
        }
        if self.stop_after == "ppr":
            return self._result(
                recorder, stage_outputs, generation,
                seed_entities=seed_trace,
                triples=retrieved_facts,
                passages=dense_passages,
                traversal_steps=iterations,
                stopping_reason="stopped_after_ppr",
                method_metadata={"recognition": recognition_metadata, "ppr_iterations": iterations},
            )
        with recorder.stage("passage_only_ranking", top_k=self.config.retrieval_top_k):
            chunk_scores = scores[self.graph.chunk_indices]
            order = np.argsort(chunk_scores)[::-1][: self.config.retrieval_top_k]
            ranked_passages = [
                {
                    "id": self.graph.node_ids[self.graph.chunk_indices[index]].removeprefix("chunk:"),
                    "score": float(chunk_scores[index]),
                    "text": "",
                }
                for index in order
            ]
        with recorder.stage("passage_hydration", mode="ppr"):
            ranked_passages = self._hydrate_passages(ranked_passages)
        ranked_trace = self._passage_trace(ranked_passages)
        stage_outputs["passage_ranking"] = ranked_trace
        logger.info(
            "HippoRAG2 sample=%s facts=%s seeds=%s ppr_iterations=%s latency=%.3fs",
            sample.sample_id, len(fact_candidates), len(seed_trace), iterations, time.perf_counter() - started,
        )
        result = self._result(
            recorder, stage_outputs, generation,
            seed_entities=seed_trace,
            triples=retrieved_facts,
            passages=ranked_trace,
            stopping_reason="stopped_after_passage_ranking" if self.stop_after else "ppr_complete",
            traversal_steps=iterations,
            method_metadata={
                "recognition": recognition_metadata,
                "ppr_iterations": iterations,
            },
        )
        return self._maybe_answer(sample, result, recorder)

    def _maybe_answer(self, sample: BenchmarkSample, result: MethodResult, recorder: StageRecorder) -> MethodResult:
        if not self.generate_answer:
            return result
        evidence = result.passages[: self.config.qa_top_k]
        retrieval_stopping_reason = result.stopping_reason
        with recorder.stage("answer_generation", passages=len(evidence)) as event:
            event["llm_calls"] = 1
            answer = self.answerer.answer(sample.text, evidence, sample.valid_labels)
            event["input_tokens"] = answer.input_tokens
            event["output_tokens"] = answer.output_tokens
            event["details"].update({
                "answer": answer.answer,
                "citations": list(answer.cited_passage_ids),
            })
        normalized = answer.answer.casefold().strip().rstrip(".")
        result.labels = [normalized] if normalized else []
        result.explanation = answer.explanation
        result.raw_response = answer.raw_response
        result.parse_status = "ok" if normalized and not answer.errors else (
            "invalid_label" if normalized else "empty"
        )
        result.llm_calls += 1
        result.input_tokens += answer.input_tokens
        result.output_tokens += answer.output_tokens
        result.errors.extend(answer.errors)
        result.stage_outputs["answer_generation"] = {
            "answer": answer.answer,
            "explanation": answer.explanation,
            "cited_passage_ids": list(answer.cited_passage_ids),
            "evidence_passage_ids": [row["passage_id"] for row in evidence],
        }
        result.method_metadata["retrieval_stopping_reason"] = retrieval_stopping_reason
        result.stopping_reason = "answer_complete" if not answer.errors else "answer_error"
        return result

    def _result(self, recorder, stage_outputs, generation, **kwargs) -> MethodResult:
        return MethodResult(
            llm_calls=(1 if self.config.recognition_mode == "llm" else 0)
            + self._last_filter_usage["calls"],
            input_tokens=generation.input_tokens + self._last_filter_usage["input_tokens"],
            output_tokens=generation.output_tokens + self._last_filter_usage["output_tokens"],
            errors=list(generation.errors),
            stage_outputs=stage_outputs,
            stages=recorder.events,
            **kwargs,
        )

    @staticmethod
    def _fact_trace(facts: list[dict], endpoints: dict | None = None) -> list[dict]:
        endpoints = endpoints or {}
        trace = []
        for rank, fact in enumerate(facts, start=1):
            subject_id, _, object_id = endpoints.get(fact["id"], (None, None, None))
            trace.append({
                "triple_id": fact["id"],
                "subject_id": subject_id,
                "subject": fact.get("subject_text"),
                "predicate_id": None,
                "predicate": fact.get("predicate"),
                "object_id": object_id,
                "object": fact.get("object_text"),
                "text": fact.get("text"),
                "score": fact["score"],
                "raw_score": fact.get("raw_score", fact["score"]),
                "rank": rank,
                "source_query_ids": list(dict.fromkeys(
                    f"fq{match['query_rank']}" for match in fact.get("matched_queries", [])
                )) or ["q1"],
                "matched_queries": fact.get("matched_queries", []),
            })
        return trace

    @staticmethod
    def _passage_trace(passages: list[dict]) -> list[dict]:
        return [
            {
                "passage_id": passage["id"],
                "document_id": None,
                "page": None,
                "source_node_id": passage["id"],
                "score": passage["score"],
                "rank": rank,
                "source_query_ids": ["q1"],
                "text": passage.get("text", ""),
            }
            for rank, passage in enumerate(passages, start=1)
        ]

    def _hydrate_passages(self, passages: list[dict]) -> list[dict]:
        if not passages:
            return passages
        ids = [passage["id"] for passage in passages]
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id,content FROM chunks WHERE id = ANY(%s)", (ids,))
            contents = {row[0]: row[1] for row in cursor}
        for passage in passages:
            passage["text"] = contents.get(passage["id"], "")
        return passages

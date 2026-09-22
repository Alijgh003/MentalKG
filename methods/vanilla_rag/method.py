from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.env import PROJECT_ENV_FILE
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.reranking import OpenAICompatibleReranker, RerankerConfig
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig
from methods.base import BenchmarkSample, MethodResult
from methods.hipporag.answering import EvidenceAnswerer
from methods.telemetry import StageRecorder


STOP_STAGES = ("passage_retrieval", "reranking", "answer_generation")


class VanillaRAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VANILLA_RAG_",
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    retrieval_top_k: int = Field(default=50, gt=0)
    rerank_top_k: int = Field(default=10, gt=0)
    qa_top_k: int = Field(default=5, gt=0)


class VanillaRAGMethod:
    name = "vanilla_rag"

    def __init__(
        self, *, generate_answer: bool = False, config_overrides: dict | None = None,
        stop_after: str | None = None,
    ):
        if stop_after is not None and stop_after not in STOP_STAGES:
            raise ValueError(f"Unknown stop stage {stop_after!r}; choose from {', '.join(STOP_STAGES)}")
        self.config = VanillaRAGConfig(**(config_overrides or {}))
        self.stop_after = stop_after
        self.generate_answer = generate_answer or stop_after == "answer_generation"
        startup = StageRecorder(scope="startup", method=self.name)
        with startup.stage("embedding_client_init"):
            self.embedder = OpenAICompatibleEmbedder(EmbeddingConfig.from_env())
        with startup.stage("milvus_connect"):
            self.store = KnowledgeGraphVectorStore(MilvusConfig.from_env())
        with startup.stage("reranker_client_init"):
            self.reranker = OpenAICompatibleReranker(RerankerConfig.from_env())
        self.answerer = None
        if self.generate_answer:
            with startup.stage("answer_program_init"):
                self.answerer = EvidenceAnswerer()
        self.startup_stages = startup.events
        self.last_stages: list[dict] = []

    def close(self) -> None:
        return None

    def run(self, sample: BenchmarkSample) -> MethodResult:
        recorder = StageRecorder(
            method=self.name,
            dataset=sample.dataset,
            split=sample.split,
            dataset_row_index=sample.dataset_row_index,
            source_row_index=sample.source_row_index,
            sample_id=sample.sample_id,
        )
        labeled_query = (
            f"{sample.text}\n\nAllowed answer labels (use these labels as the task label set): "
            f"{', '.join(sample.valid_labels)}"
        )
        with recorder.stage("passage_query_embedding") as event:
            event["embedding_calls"] = 1
            vector = self.embedder.embed(
                [labeled_query], dimension=self.store.config.source_chunk_dimension
            )[0].tolist()
        with recorder.stage(
            "passage_retrieval", limit=self.config.retrieval_top_k
        ) as event:
            event["retrieval_calls"] = 1
            hits = self.store.search(
                self.store.config.source_chunk_collection,
                [vector],
                limit=self.config.retrieval_top_k,
                output_fields=("text",),
            )[0]
            event["details"]["hits"] = len(hits)
        dense = [
            {
                "passage_id": str(hit["id"]),
                "text": hit.get("entity", {}).get("text", ""),
                "dense_score": float(hit.get("distance", 0.0)),
                "score": float(hit.get("distance", 0.0)),
                "rank": rank,
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        stage_outputs = {"passage_retrieval": dense}
        if self.stop_after == "passage_retrieval":
            return self._result(recorder, stage_outputs, dense, "stopped_after_passage_retrieval")

        with recorder.stage(
            "reranking", candidates=len(dense), top_n=self.config.rerank_top_k
        ) as event:
            event["retrieval_calls"] = 1
            reranked_hits = self.reranker.rerank(
                labeled_query,
                [row["text"] for row in dense],
                top_n=self.config.rerank_top_k,
            )
            event["details"]["hits"] = len(reranked_hits)
        reranked = [
            {
                **dense[hit.index],
                "dense_rank": dense[hit.index]["rank"],
                "rank": rank,
                "score": hit.score,
                "rerank_score": hit.score,
            }
            for rank, hit in enumerate(reranked_hits, start=1)
        ]
        stage_outputs["reranking"] = reranked
        if self.stop_after == "reranking":
            return self._result(recorder, stage_outputs, reranked, "stopped_after_reranking")

        result = self._result(recorder, stage_outputs, reranked, "reranking_complete")
        if not self.generate_answer:
            return result
        evidence = reranked[: self.config.qa_top_k]
        with recorder.stage("answer_generation", passages=len(evidence)) as event:
            event["llm_calls"] = 1
            answer = self.answerer.answer(sample.text, evidence, sample.valid_labels)
            event["input_tokens"] = answer.input_tokens
            event["output_tokens"] = answer.output_tokens
            event["details"].update({
                "answer": answer.answer,
                "citations": list(answer.cited_passage_ids),
                "valid_labels": list(sample.valid_labels),
            })
        normalized = answer.answer.casefold().strip().rstrip(".")
        result.labels = [normalized] if normalized else []
        result.explanation = answer.explanation
        result.raw_response = answer.raw_response
        result.parse_status = "ok" if normalized and not answer.errors else (
            "invalid_label" if normalized else "empty"
        )
        result.llm_calls = 1
        result.input_tokens = answer.input_tokens
        result.output_tokens = answer.output_tokens
        result.errors.extend(answer.errors)
        result.stage_outputs["answer_generation"] = {
            "answer": answer.answer,
            "explanation": answer.explanation,
            "cited_passage_ids": list(answer.cited_passage_ids),
            "evidence_passage_ids": [row["passage_id"] for row in evidence],
        }
        result.stages = recorder.events
        result.stopping_reason = "answer_complete" if not answer.errors else "answer_error"
        self.last_stages = recorder.events
        return result

    def _result(self, recorder, stage_outputs, passages, stopping_reason) -> MethodResult:
        self.last_stages = recorder.events
        return MethodResult(
            passages=passages,
            stage_outputs=stage_outputs,
            stages=recorder.events,
            stopping_reason=stopping_reason,
            method_metadata={
                "retrieval": "dense",
                "reranker_model": self.reranker.config.model,
            },
        )

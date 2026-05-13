import pandas as pd
import dspy
from dspy import BootstrapFewShot
import time
import logging
from huey import RedisHuey
from config import settings
from signatures import (
    entity_example,
    entity_extraction,
    relation_examples,
    relation_extraction,
)
from tasks.info_extraction_store import store_relation_results, store_entity_results

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("entity_relation_tasks.log"),
        logging.StreamHandler(),
    ],
)

# ---------------------------
# Huey instances
# ---------------------------
huey_extraction = RedisHuey(
    name="entity_relation_tasks",
    host="192.168.1.204",
    port=6379,
    db=0,
    password="yourpassword",
)

# ---------------------------
# dspy / LLM / COT setup
# ---------------------------
lm = dspy.LM(
    model=settings.llm_model,
    api_base=settings.llm_api_base,
    api_key=settings.llm_api_key,
    timeout=settings.llm_timeout,
)

dspy.settings.configure(lm=lm, temperature=0.0)
optimizer = BootstrapFewShot()

entity_extraction_cot_child = dspy.ChainOfThought(
    entity_extraction.DisorderEntityExtractor
)
entity_extraction_cot = optimizer.compile(
    entity_extraction_cot_child,
    trainset=[
        entity_example.case_study_para1_entities,
        entity_example.case_study_para2_entities,
        entity_example.example_entities,
    ],
)

relation_extraction_cot_child = dspy.ChainOfThought(
    relation_extraction.DisorderRelationExtractor
)
relation_extraction_cot = optimizer.compile(
    relation_extraction_cot_child,
    trainset=[
        relation_examples.case_study_para1_relations,
        relation_examples.case_study_para2_relations,
        relation_examples.example_relations,
    ],
)


# ---------------------------
# Huey Tasks with logging
# ---------------------------
@huey_extraction.task(retries=3, retry_delay=10)
def call_entity_extraction(content, _path, node_id):
    logging.info(f"Starting entity extraction for node_id={node_id}")
    start_time = time.time()
    try:
        pred_result = entity_extraction_cot(text=content, context_hierarchy=_path)
        elapsed = time.time() - start_time
        result = {}
        result["execution_time"] = elapsed
        result["node_id"] = node_id
        logging.info(
            f"Entity extraction finished for node_id={node_id} in {elapsed:.2f}s"
        )
        result["entities"] = pred_result.entities
        result["passage_type"] = pred_result.passage_type

        store_entity_results(result)

        if pred_result.get("alright", False):
            logging.info(f"Triggering relation extraction for node_id={node_id}")
            extract_relation(
                entities=pred_result.entities,
                content=content,
                _path=_path,
                node_id=node_id,
                passage_type=pred_result.passage_type,
            )
        return result

    except Exception as e:
        logging.error(f"Entity extraction failed for node_id={node_id}: {str(e)}")
        return {
            "node_id": node_id,
            "error": str(e),
            "execution_time": time.time() - start_time,
        }


@huey_extraction.task(retries=3, retry_delay=10)
def extract_relation(entities, content, _path, node_id, passage_type):
    logging.info(f"Starting relation extraction for node_id={node_id}")
    start_time = time.time()
    try:
        pred_rel_result = relation_extraction_cot(
            text=content,
            entities=entities,
            passage_type=passage_type,
            context_section=_path,
        )
        elapsed = time.time() - start_time
        rel_result = {}
        rel_result["execution_time"] = elapsed
        rel_result["node_id"] = node_id
        rel_result["triples"] = pred_rel_result.triples
        logging.info(
            f"Relation extraction finished for node_id={node_id} in {elapsed:.2f}s"
        )

        store_relation_results(rel_result)
        return rel_result

    except Exception as e:
        logging.error(f"Relation extraction failed for node_id={node_id}: {str(e)}")
        return {
            "node_id": node_id,
            "error": str(e),
            "execution_time": time.time() - start_time,
        }


# ---------------------------
# Schedule tasks with logging
# ---------------------------
def schedule_entity_extraction_tasks(df_path: str):
    logging.info(f"Loading DataFrame from {df_path}")
    df = pd.read_json(df_path, lines=True)
    df = df[df["node_type_in_tree"] == "leaf"]
    df = df.iloc[:100]
    for _, row in df.iterrows():
        content = row["content"]
        _path = row["_path"]
        node_id = row["node_id"]
        logging.info(f"Scheduling entity extraction task for node_id={node_id}")
        call_entity_extraction(content, _path, node_id)


# ---------------------------
# Usage
# ---------------------------
schedule_entity_extraction_tasks("books/dsm-tree/dsm5_selected_chapters_tree.jsonl")

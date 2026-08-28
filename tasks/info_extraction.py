import argparse
import logging
import time
from functools import lru_cache

import pandas as pd
from huey import RedisHuey

from config import settings
from kg_pipeline.extraction import EntityExtraction, build_dspy_extractor
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

@lru_cache(maxsize=1)
def get_extractor():
    """Create the costly DSPy programs once per Huey worker process."""
    return build_dspy_extractor(settings)


# ---------------------------
# Huey Tasks with logging
# ---------------------------
@huey_extraction.task(retries=3, retry_delay=10)
def call_entity_extraction(content, _path, node_id):
    logging.info(f"Starting entity extraction for node_id={node_id}")
    start_time = time.time()
    try:
        extraction = get_extractor().extract_entities(content, _path, node_id)
        elapsed = time.time() - start_time
        result = {}
        result["execution_time"] = elapsed
        result["node_id"] = node_id
        logging.info(
            f"Entity extraction finished for node_id={node_id} in {elapsed:.2f}s"
        )
        result["entities"] = extraction.entities
        result["passage_type"] = extraction.passage_type

        store_entity_results(result)

        if extraction.accepted:
            logging.info(f"Triggering relation extraction for node_id={node_id}")
            extract_relation(
                entities=extraction.entities,
                content=content,
                _path=_path,
                node_id=node_id,
                passage_type=extraction.passage_type,
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
        entity_result = EntityExtraction(
            node_id=node_id,
            entities=entities,
            passage_type=passage_type,
            accepted=True,
        )
        relation_extraction = get_extractor().extract_relations(content, _path, entity_result)
        elapsed = time.time() - start_time
        rel_result = {}
        rel_result["execution_time"] = elapsed
        rel_result["node_id"] = node_id
        rel_result["triples"] = relation_extraction.triples
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
def schedule_entity_extraction_tasks(df_path: str, limit: int | None = None):
    """Enqueue leaf passages; importing this module never schedules work."""
    logging.info(f"Loading DataFrame from {df_path}")
    df = pd.read_json(df_path, lines=True)
    df = df[df["node_type_in_tree"] == "leaf"]
    if limit is not None:
        df = df.iloc[:limit]
    for _, row in df.iterrows():
        content = row["content"]
        _path = row["_path"]
        node_id = row["node_id"]
        logging.info(f"Scheduling entity extraction task for node_id={node_id}")
        call_entity_extraction(content, _path, node_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enqueue DSM leaf nodes for KG extraction")
    parser.add_argument("--nodes", default="books/dsm-tree/dsm5_selected_chapters_tree.jsonl")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    schedule_entity_extraction_tasks(arguments.nodes, arguments.limit)

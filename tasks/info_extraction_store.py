import logging
import os
from pathlib import Path

import pandas as pd
from huey import RedisHuey

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("storage_tasks.log"), logging.StreamHandler()],
)

# ---------------------------
# Huey Storage Instance
# ---------------------------
huey_storage = RedisHuey(
    name="storage_tasks",
    host="192.168.1.204",
    port=6379,
    db=0,
    password="yourpassword",
)

ARTIFACT_DIRECTORY = Path(
    os.getenv("KG_ARTIFACT_DIRECTORY", "books/dsm5-KG/extraction_artifacts")
)


def _write_artifact(kind: str, node_id: str, records: list[dict]) -> Path:
    """Persist worker output beneath the repository instead of a machine-specific path."""
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIRECTORY / f"{kind}_{node_id}.json"
    pd.DataFrame(records).to_json(path, orient="records")
    return path


# ---------------------------
# Storage Tasks with Logging
# ---------------------------
@huey_storage.task(retries=2, retry_delay=5)
def store_entity_results(result):
    node_id = result.get("node_id")
    logging.info(f"Storing entity results for node_id={node_id}")
    try:
        entities = result.get("entities", [])
        _write_artifact("entities", node_id, entities)
        logging.info(
            f"Successfully stored entity results for node_id={node_id} ({len(entities)} entities)"
        )
    except Exception as e:
        logging.error(f"Failed to store entity results for node_id={node_id}: {e}")


@huey_storage.task(retries=2, retry_delay=5)
def store_relation_results(result):
    node_id = result.get("node_id")
    logging.info(f"Storing relation results for node_id={node_id}")
    try:
        triples = result.get("triples", [])
        _write_artifact("relations", node_id, triples)
        logging.info(
            f"Successfully stored relation results for node_id={node_id} ({len(triples)} triples)"
        )
    except Exception as e:
        logging.error(f"Failed to store relation results for node_id={node_id}: {e}")

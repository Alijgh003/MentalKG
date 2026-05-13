import logging
from huey import RedisHuey
import pandas as pd

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


# ---------------------------
# Storage Tasks with Logging
# ---------------------------
@huey_storage.task(retries=2, retry_delay=5)
def store_entity_results(result):
    node_id = result.get("node_id")
    logging.info(f"Storing entity results for node_id={node_id}")
    try:
        entities = result.get("entities", [])
        pd.DataFrame(entities).to_json(
            f"~/work/dsm5-KG/entities_{node_id}.json", orient="records"
        )
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
        pd.DataFrame(triples).to_json(
            f"~/work/dsm5-KG/relations_{node_id}.json", orient="records"
        )
        logging.info(
            f"Successfully stored relation results for node_id={node_id} ({len(triples)} triples)"
        )
    except Exception as e:
        logging.error(f"Failed to store relation results for node_id={node_id}: {e}")

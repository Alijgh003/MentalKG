from huey import RedisHuey
import pandas as pd

# ---------------------------
# Storage Tasks
# ---------------------------

huey_storage = RedisHuey(
    name="storage_tasks",
    host="192.168.1.24",
    port=6379,
    db=0,
    password="yourpassword",
)


@huey_storage.task(retries=2, retry_delay=5)
def store_entity_results(result):
    """
    Store entity extraction results into DB or filesystem.
    """
    try:
        node_id = result.get("node_id")
        # Example: save to JSON file per node
        pd.DataFrame(result.get("entities", [])).to_json(
            f"./entities_{node_id}.json", orient="records"
        )
    except Exception as e:
        print(f"Failed to store entity results for node {node_id}: {e}")


@huey_storage.task(retries=2, retry_delay=5)
def store_relation_results(result):
    """
    Store relation extraction results into DB or filesystem.
    """
    try:
        node_id = result.get("node_id")
        pd.DataFrame(result.get("triples", [])).to_json(
            f"./relations_{node_id}.json", orient="records"
        )
    except Exception as e:
        print(f"Failed to store relation results for node {node_id}: {e}")

import pandas as pd
import dspy
import time
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
# Huey instances
# ---------------------------
huey_extraction = RedisHuey(
    name="entity_relation_tasks",
    host="192.168.1.24",
    port=6379,
    db=0,  # Redis database number
    password="yourpassword",  # set if your Redis requires password
)

# ---------------------------
# dspy / LLM / COT setup
# ---------------------------
# Initialize dspy LLM config

dspy_obj = dspy.LM(
    model=settings.llm_model,
    api_base=settings.llm_api_base,
    api_key=settings.llm_api_key,
    timeout=settings.llm_timeout,
)


# Entity extraction COT with signature
entity_extraction_cot = dspy.COT(
    lm=dspy_obj,
    signature=entity_extraction.DisorderEntityExtractor,
    few_shot_examples=[
        # few-shot examples for entity extraction
        entity_example.case_study_para1_entities,
        entity_example.case_study_para2_entities,
        entity_example.example_entities,
    ],
)

# Relation extraction COT with signature
relation_extraction_cot = dspy.COT(
    lm=dspy_obj,
    signature=relation_extraction.DisorderRelationExtractor,
    few_shot_examples=[
        relation_examples.case_study_para1_relations,
        relation_examples.case_study_para2_relations,
        relation_examples.example_relations,
    ],
)

# ---------------------------
# Huey Tasks
# ---------------------------


@huey_extraction.task(retries=3, retry_delay=10)
def call_entity_extraction(content, _path, node_id):
    """
    Calls the entity extraction COT for a given text passage.
    """
    start_time = time.time()
    try:
        result = entity_extraction_cot(text=content, context_hierarchy=_path)
        elapsed = time.time() - start_time
        # Add execution time
        result["execution_time"] = elapsed
        result["node_id"] = node_id

        # Store entity extraction results
        store_entity_results(result)

        # If extract    ion is alright, trigger relation extraction
        if result.get("alright", False):
            extract_relation(
                entities=result["entities"],
                content=content,
                _path=_path,
                node_id=node_id,
                passage_type=result["passage_type"],  # <- use this
            )
        return result

    except Exception as e:
        return {
            "node_id": node_id,
            "error": str(e),
            "execution_time": time.time() - start_time,
        }


@huey_extraction.task(retries=3, retry_delay=10)
def extract_relation(entities, content, _path, node_id, passage_type):
    start_time = time.time()
    try:
        rel_result = relation_extraction_cot(
            text=content,
            entities=entities,
            passage_type=passage_type,  # <- use the entity extraction output
            context_section=_path,
        )
        elapsed = time.time() - start_time
        rel_result["execution_time"] = elapsed
        rel_result["node_id"] = node_id

        # store results
        store_relation_results(rel_result)
        return rel_result

    except Exception as e:
        return {
            "node_id": node_id,
            "error": str(e),
            "execution_time": time.time() - start_time,
        }


# ---------------------------
# Example: Read a pandas DF and schedule tasks
# ---------------------------
def schedule_entity_extraction_tasks(df_path: str):
    df = pd.read_json(df_path, lines=True)
    for _, row in df.iterrows():
        content = row["content"]
        _path = row["_path"]
        node_id = row["node_id"]
        call_entity_extraction(content, _path, node_id)


# ---------------------------
# Usage
# ---------------------------
schedule_entity_extraction_tasks("books/dsm-tree/dsm5_selected_chapters_tree.jsonl")

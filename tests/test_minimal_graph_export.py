import unittest
import uuid

from kg_pipeline.minimal_graph_export import chunk_content, unresolved_entity_id


class MinimalGraphExportTests(unittest.TestCase):
    def test_chunk_content_is_content_then_hierarchy(self):
        self.assertEqual(chunk_content(" body ", " root > leaf "), "body\n\nroot > leaf")
        self.assertEqual(chunk_content("body", None), "body")

    def test_unresolved_ids_are_deterministic_and_role_scoped(self):
        relation_id = uuid.uuid4()
        self.assertEqual(
            unresolved_entity_id(relation_id, "subject"),
            unresolved_entity_id(relation_id, "subject"),
        )
        self.assertNotEqual(
            unresolved_entity_id(relation_id, "subject"),
            unresolved_entity_id(relation_id, "object"),
        )


if __name__ == "__main__":
    unittest.main()

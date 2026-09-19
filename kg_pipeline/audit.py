"""Small, read-only consistency report for an imported DSM knowledge graph."""

from __future__ import annotations

import json


AUDIT_SQL = {
    "table_counts": """
        SELECT jsonb_build_object(
          'documents', (SELECT count(*) FROM documents),
          'pages', (SELECT count(*) FROM pages),
          'nodes', (SELECT count(*) FROM nodes),
          'selected_leaf_nodes', (SELECT count(*) FROM nodes WHERE is_selected AND tree_role='leaf' AND NOT is_derived),
          'boundary_nodes', (SELECT count(*) FROM nodes WHERE is_derived),
          'entities', (SELECT count(*) FROM entity_mentions),
          'relations', (SELECT count(*) FROM relation_mentions),
          'linked_endpoints', (SELECT count(*) FROM relation_entity_links),
          'endpoint_issues', (SELECT count(*) FROM relation_ingestion_issues)
        )
    """,
    "extraction_status": """
        SELECT extraction_set, status, has_entities, has_relations, count(*)
        FROM node_extraction_status GROUP BY extraction_set, status, has_entities, has_relations
        ORDER BY 1, 2, 3, 4
    """,
    "endpoint_issues": """
        SELECT issue_kind, endpoint_role, count(*)
        FROM relation_ingestion_issues GROUP BY issue_kind, endpoint_role ORDER BY 1, 2
    """,
    "top_entity_types": """
        SELECT coalesce(entity_type, '<missing>'), count(*)
        FROM entity_mentions GROUP BY 1 ORDER BY 2 DESC LIMIT 20
    """,
    "top_predicates": """
        SELECT predicate, count(*) FROM relation_mentions
        GROUP BY predicate ORDER BY 2 DESC LIMIT 25
    """,
}


def audit_database(connection) -> dict:
    report = {}
    with connection.cursor() as cursor:
        for name, query in AUDIT_SQL.items():
            cursor.execute(query)
            rows = cursor.fetchall()
            report[name] = rows[0][0] if name == "table_counts" else rows
    return report


def print_audit(connection) -> None:
    print(json.dumps(audit_database(connection), ensure_ascii=False, indent=2, default=str))

import dspy


class DisorderRelationExtractor(dspy.Signature):
    """Extract relations (triples) from a DSM-5 passage, given extracted entities.

    You are an expert in knowledge graph construction for psychodiagnostic manuals.

    Rules:
    1. Use ANY meaningful predicate name, but **normalize it to lowercase** (e.g., "has_symptom", "manifests_as", "differential_diagnosis_with").
    2. Entity types must also be **lowercase** (e.g., "disorder", "symptom", "criterion", "duration", "specifier", "patient", "concept", "code").
    3. Subject and object must match entity names exactly as provided (case-sensitive original, but the relation will be stored with lowercase normalized predicate).
    4. If the text implies a clear semantic relationship, create a triple. Be comprehensive but avoid hallucination.
    5. For case studies, add metadata: {"source_type": "case_study"}.
    6. For fine‑print lists, add metadata: {"source_type": "fine_print", "order": N}.
    7. If passage_type is 'case_study_narrative' or 'case_study_evaluation', add metadata {"source_type": "case_study"} to all triples.
    8. If passage_type is 'fine_print', add metadata {"source_type": "fine_print", "list_item": true} for bullet points.
    9. If passage_type is 'diagnostic_criteria', prioritize HAS_CRITERION, MIN_DURATION relations.

    Output format: list of dicts, each with keys: subject, predicate, object, metadata (optional dict).
    """

    text: str = dspy.InputField(desc="Original DSM-5 text passage")
    entities: list[dict] = dspy.InputField(
        desc="Entities extracted from the same passage. Each dict has 'name' (string) and 'type' (string, lowercase normalized)."
    )
    context_section: str = dspy.InputField(
        desc="Section name to inform extraction (e.g., 'diagnostic criteria', 'coding notes', 'case study')"
    )
    passage_type: str = dspy.InputField(desc="From entity extractor")

    triples: list[dict] = dspy.OutputField(
        desc="List of triples with subject, predicate (lowercase), object, and optional metadata"
    )

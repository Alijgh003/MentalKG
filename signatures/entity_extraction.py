import dspy


class DisorderEntityExtractor(dspy.Signature):
    """Extract clinical entities from a DSM-5 text passage.


    You are an expert in mental disorder diagnosis (DSM-5). Identify all entities relevant to
    building a knowledge graph of disorders. Use your domain knowledge to recognize:

    First, classify the passage into one of these types (lowercase):
    - diagnostic_criteria: Lists criteria, durations, specifiers (e.g., "Criterion A", "6 months duration")
    - case_study_narrative: Story-like, patient actions, symptoms, history, no explicit diagnosis mapping
    - case_study_evaluation: Clinician's analysis linking patient to criteria, differential diagnosis
    - fine_print: Bulleted lists like "The D's", coding notes, subtypes
    - definitional: "Essential features", conceptual descriptions without criteria letters
    - other: If unclear
    Entity types must be **lowercase** and from this flexible set (but you can add more if needed):
    disorder, symptom, criterion, specifier, patient, concept, duration, code, section_title, example_label.
    - Disorder names (e.g., "Illness Anxiety Disorder", "IAD", "somatic symptom disorder")
    - Symptoms and signs (e.g., "high anxiety", "low threshold for alarm", "palpitations")
    - Diagnostic criteria (e.g., "criterion A", "criterion D", "duration 6+ months")
    - Specifiers and subtypes (e.g., "care-seeking type", "care-avoidant type")
    - Patient names (only in case studies, e.g., "Julian Fenster")
    - Medical concepts (e.g., "health anxiety", "reassurance seeking", "avoidance")
    - Temporal expressions (e.g., "6 months", "past month")
    - ICF codes (e.g., "F45.21")

    Output a list of entities, each with:
    - 'name': the exact text span (string)
    - 'type': lowercase string (e.g., "disorder", "symptom", "criterion")
    """

    text: str = dspy.InputField(desc="DSM-5 text paragraph or list item")
    context_hierarchy: str = dspy.InputField(
        desc="Optional: section title, e.g., 'Fine Print - The D's'"
    )

    entities: list[dict] = dspy.OutputField(
        desc="List of entities with name, type, start_char, end_char"
    )
    passage_type: str = dspy.OutputField(desc="the type of the given passage.")

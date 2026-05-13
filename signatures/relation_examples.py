import dspy
from signatures.entity_example import (
    example_entities,
    case_study_para1_entities,
    case_study_para2_entities,
)

example_relations = dspy.Example(
    text="""DSM-5-TR criteria for somatic symptom disorder (SSD) require only a single somatic symptom and 6 months duration, but that symptom (or symptoms) must cause distress or markedly impair the person’s functioning. Nonetheless, the classic patient has a pattern of multiple physical and emotional symptoms that can affect various (often many) areas of the body, with the pattern of concerns lasting far beyond the minimum 6 months. The symptom areas involved can include pain symptoms, problems with breathing or heartbeat, abdominal complaints, and menstrual disorders. Functional neurological symptoms (apparent body malfunctioning such as paralysis or blindness that has no anatomical or physiological cause) may also be encountered. Treatment that usually helps symptoms caused by actual physical disease is often ineffective for these patients.""",
    entities=example_entities.entities,  # now without char indices
    passage_type="definitional",
    context_section="Diagnostic criteria and symptom description",
    triples=[
        {
            "subject": "somatic symptom disorder",
            "predicate": "has_criterion",
            "object": "single somatic symptom",
            "metadata": {"order": 1},
        },
        {
            "subject": "somatic symptom disorder",
            "predicate": "min_duration",
            "object": "6 months duration",
        },
        {
            "subject": "somatic symptom disorder",
            "predicate": "has_criterion",
            "object": "markedly impair the person’s functioning",
            "metadata": {"order": 2},
        },
        {
            "subject": "somatic symptom disorder",
            "predicate": "manifests_as",
            "object": "multiple physical and emotional symptoms",
        },
        {
            "subject": "somatic symptom disorder",
            "predicate": "manifests_as",
            "object": "pain symptoms",
        },
        {
            "subject": "somatic symptom disorder",
            "predicate": "manifests_as",
            "object": "problems with breathing or heartbeat",
        },
        {
            "subject": "functional neurological symptoms",
            "predicate": "exhibits",
            "object": "paralysis",
            "metadata": {"example": True},
        },
        {
            "subject": "functional neurological symptoms",
            "predicate": "exhibits",
            "object": "blindness",
            "metadata": {"example": True},
        },
        {
            "subject": "treatment that usually helps symptoms caused by actual physical disease",
            "predicate": "has_effect_on",
            "object": "somatic symptom disorder",
            "metadata": {"effect": "ineffective"},
        },
    ],
)

case_study_para1_relations = dspy.Example(
    text=""""Wow! My chart must be 2 inches thick." Julian Fenster is checking in for his third emergency room visit in the past month. "That's just Volume 3," the nurse tells him. At age 24, Julian lives with his mother and a teenage sister. Years ago, he enrolled at a college several hundred miles away. After only a semester, he moved back home. "I didn't want to be that far from my doctors," he explains. "When you're trying to prevent heart disease, you can't be too careful." With a practiced hand, he adjusts the blood pressure cuff around his upper arm.""",
    entities=case_study_para1_entities["entities"],
    passage_type="case_study_narrative",
    context_section="case study narrative",
    triples=[
        {
            "subject": "Julian Fenster",
            "predicate": "exhibits_behavior",
            "object": "third emergency room visit in the past month",
            "metadata": {"source_type": "case_study", "paragraph": 1},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "lives_with",
            "object": "his mother and a teenage sister",
            "metadata": {"source_type": "case_study", "paragraph": 1},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "performed_action",
            "object": "enrolled at a college several hundred miles away",
            "metadata": {"source_type": "case_study", "paragraph": 1},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "performed_action",
            "object": "moved back home",
            "metadata": {"source_type": "case_study", "paragraph": 1},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "expresses_concern_about",
            "object": "prevent heart disease",
            "metadata": {"source_type": "case_study", "paragraph": 1},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "performs_action",
            "object": "adjusts the blood pressure cuff",
            "metadata": {"source_type": "case_study", "paragraph": 1},
        },
    ],
)

case_study_para2_relations = dspy.Example(
    text="""When Julian was a young teenager, his dad died. "His death was self-inflicted," Julian points out. "He'd had rheumatic fever as a child, which gave him an enlarged heart. And the only thing he ever exercised was his right to eat anything fried, including Twinkies. And he smoked—he was a proud two-pack-a-day man. Look where that got him." None of these health risks apply to Julian, who is nothing if not careful about what he puts into his body. He has spent hours searching the Internet for information on diet, and he once attended a lecture by Dean Ornish. "I've followed a plant-based diet ever since," Julian said. "I'm especially keen on tofu. And broccoli." Julian has never complained much about having symptoms—just the odd palpitation, maybe "hot flushes" on an especially humid day. "I don't feel bad," he explains. "I just feel scared." This time, he's heard a report on NPR about young people with heart disease. It startled him so much he dropped the dish he had been putting into the cupboard. Without even cleaning up the mess, he caught the next bus to the ER.""",
    entities=case_study_para2_entities["entities"],
    passage_type="case_study_narrative",
    context_section="case study narrative",
    triples=[
        {
            "subject": "Julian Fenster",
            "predicate": "experienced_event",
            "object": "dad died",
            "metadata": {"source_type": "case_study", "paragraph": 2},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "reports_symptom",
            "object": "palpitation",
            "metadata": {"source_type": "case_study", "paragraph": 2},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "reports_symptom",
            "object": "hot flushes",
            "metadata": {"source_type": "case_study", "paragraph": 2},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "feels_emotion",
            "object": "scared",
            "metadata": {"source_type": "case_study", "paragraph": 2},
        },
        {
            "subject": "report on NPR about young people with heart disease",
            "predicate": "triggers",
            "object": "dropped the dish",
            "metadata": {"source_type": "case_study", "paragraph": 2},
        },
        {
            "subject": "Julian Fenster",
            "predicate": "responds_by",
            "object": "caught the next bus to the ER",
            "metadata": {"source_type": "case_study", "paragraph": 2},
        },
    ],
)

case_study_para2_relations = case_study_para2_relations.with_inputs(
    "text", "entities", "passage_type", "context_section"
)
case_study_para1_relations = case_study_para1_relations.with_inputs(
    "text", "entities", "passage_type", "context_section"
)
example_relations = example_relations.with_inputs(
    "text", "entities", "context_section", "passage_type"
)

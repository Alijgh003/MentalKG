import dspy

example_entities = dspy.Example(
    text="""DSM-5-TR criteria for somatic symptom disorder (SSD) require only a single somatic symptom and 6 months duration, but that symptom (or symptoms) must cause distress or markedly impair the person’s functioning. Nonetheless, the classic patient has a pattern of multiple physical and emotional symptoms that can affect various (often many) areas of the body, with the pattern of concerns lasting far beyond the minimum 6 months. The symptom areas involved can include pain symptoms, problems with breathing or heartbeat, abdominal complaints, and menstrual disorders. Functional neurological symptoms (apparent body malfunctioning such as paralysis or blindness that has no anatomical or physiological cause) may also be encountered. Treatment that usually helps symptoms caused by actual physical disease is often ineffective for these patients.""",
    context_hierarchy="Chapter 8 -> Somatic Symptom and Related Disorders -> Introduction -> F45.1 Somatic Symptom Disorder",
    passage_type="definitional",
    entities=[
        {"name": "somatic symptom disorder", "type": "disorder"},
        {"name": "SSD", "type": "disorder"},
        {"name": "single somatic symptom", "type": "criterion"},
        {"name": "6 months duration", "type": "duration"},
        {"name": "distress", "type": "symptom"},
        {"name": "markedly impair the person’s functioning", "type": "criterion"},
        {"name": "multiple physical and emotional symptoms", "type": "symptom"},
        {"name": "pain symptoms", "type": "symptom"},
        {"name": "problems with breathing or heartbeat", "type": "symptom"},
        {"name": "abdominal complaints", "type": "symptom"},
        {"name": "menstrual disorders", "type": "symptom"},
        {"name": "functional neurological symptoms", "type": "symptom"},
        {"name": "paralysis", "type": "symptom"},
        {"name": "blindness", "type": "symptom"},
        {"name": "no anatomical or physiological cause", "type": "concept"},
        {"name": "ineffective", "type": "concept"},
    ],
    alright=True,
)

case_study_para1_entities = dspy.Example(
    text=""""Wow! My chart must be 2 inches thick." Julian Fenster is checking in for his third emergency room visit in the past month. "That's just Volume 3," the nurse tells him. At age 24, Julian lives with his mother and a teenage sister. Years ago, he enrolled at a college several hundred miles away. After only a semester, he moved back home. "I didn't want to be that far from my doctors," he explains. "When you're trying to prevent heart disease, you can't be too careful." With a practiced hand, he adjusts the blood pressure cuff around his upper arm.""",
    context_hierarchy="chapter 8 -> somatic symptom and related disorders -> illness anxiety disorder -> case study: julian fenster",
    passage_type="case_study_narrative",
    alright=True,
    entities=[
        {"name": "Julian Fenster", "type": "patient"},
        {
            "name": "third emergency room visit in the past month",
            "type": "behavior",
        },
        {"name": "age 24", "type": "demographic"},
        {
            "name": "lives with his mother and a teenage sister",
            "type": "social_context",
        },
        {
            "name": "enrolled at a college several hundred miles away",
            "type": "behavior",
        },
        {"name": "moved back home", "type": "behavior"},
        {"name": "prevent heart disease", "type": "concern"},
        {"name": "adjusts the blood pressure cuff", "type": "behavior"},
    ],
)

case_study_para2_entities = dspy.Example(
    text="""When Julian was a young teenager, his dad died. "His death was self-inflicted," Julian points out. "He'd had rheumatic fever as a child, which gave him an enlarged heart. And the only thing he ever exercised was his right to eat anything fried, including Twinkies. And he smoked—he was a proud two-pack-a-day man. Look where that got him." None of these health risks apply to Julian, who is nothing if not careful about what he puts into his body. He has spent hours searching the Internet for information on diet, and he once attended a lecture by Dean Ornish. "I've followed a plant-based diet ever since," Julian said. "I'm especially keen on tofu. And broccoli." Julian has never complained much about having symptoms—just the odd palpitation, maybe "hot flushes" on an especially humid day. "I don't feel bad," he explains. "I just feel scared." This time, he's heard a report on NPR about young people with heart disease. It startled him so much he dropped the dish he had been putting into the cupboard. Without even cleaning up the mess, he caught the next bus to the ER.""",
    context_hierarchy="chapter 8 -> somatic symptom and related disorders -> illness anxiety disorder -> case study: julian fenster",
    alright=True,
    passage_type="case_study_narrative",
    entities=[
        {"name": "Julian Fenster", "type": "patient"},
        {"name": "dad died", "type": "life_event"},
        {"name": "self-inflicted", "type": "life_event"},
        {"name": "rheumatic fever", "type": "medical_history"},
        {"name": "enlarged heart", "type": "medical_history"},
        {
            "name": "searches the Internet for information on diet",
            "type": "behavior",
        },
        {"name": "plant-based diet", "type": "behavior"},
        {"name": "palpitation", "type": "symptom"},
        {"name": "hot flushes", "type": "symptom"},
        {"name": "scared", "type": "emotional_state"},
        {
            "name": "report on NPR about young people with heart disease",
            "type": "trigger",
        },
        {"name": "dropped the dish", "type": "reaction"},
        {"name": "caught the next bus to the ER", "type": "behavior"},
    ],
)

case_study_para2_entities = case_study_para2_entities.with_inputs(
    "text", "context_hierarchy"
)
case_study_para1_entities = case_study_para1_entities.with_inputs(
    "text", "context_hierarchy"
)
example_entities = example_entities.with_inputs("text", "context_hierarchy")

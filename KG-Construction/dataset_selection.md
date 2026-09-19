# Dataset Selection for the DSM-Derived KG

## Decision

The primary benchmark should test tasks whose labels and explanations can be
supported directly by the DSM-derived graph. The current IMHI tasks remain
useful as transfer or knowledge-source-mismatch experiments, but they should not
be the main evidence for the value of a DSM-derived KG.

## Core datasets

### 1. ReDSM5

- **Primary tasks:** multi-label detection of the nine DSM-5 depression symptoms
  and evidence-grounded explanation generation.
- **Why it fits:** all 1,484 long-form Reddit posts were annotated at sentence
  level by a licensed psychologist; every symptom label has a concise rationale
  grounded in DSM-5 methodology, and 392 posts are negative contrasts.
- **Labels:** depressed mood, anhedonia, appetite change, sleep issues,
  psychomotor alteration, fatigue, worthlessness, cognitive issues, and suicidal
  thoughts.
- **Evaluation value:** it directly tests symptom retrieval, evidence selection,
  multi-label prediction, and explanation faithfulness against expert rationales.
- **Access:** the full dataset is gated on Hugging Face and requires accepting
  the terms and emailing the agreement form; a small paraphrased preview and the
  code are public.
- **Sources:** [paper](https://arxiv.org/abs/2508.03399),
  [repository](https://github.com/eliseobao/redsm5),
  [dataset page](https://huggingface.co/datasets/irlab-udc/redsm5).

### 2. PsySym

- **Primary tasks:** multi-label symptom relevance and symptom status inference
  from social-media sentences; its downstream user-level disorder detection set
  is an optional additional task.
- **Why it fits:** PsySym contains 8,554 sentences annotated for 38 standardized
  symptom classes associated with seven disorders: depression, anxiety, ADHD,
  bipolar disorder, OCD, PTSD, and eating disorder. Its annotation targets were
  built mainly from DSM-5 criteria, supplemented by clinical questionnaires.
- **Splits and controls:** the symptom data has train/validation/test splits of
  5:1:4 and the work also collected 83,779 control sentences. The associated
  disease-detection resource contains 5,624 diagnosed and 20,981 control users.
- **Evaluation value:** relevance and status are especially suitable for testing
  negation, uncertainty, semantic entity linking, shared symptoms across
  disorders, and transparent graph paths.
- **Access:** code and the schema of the data are public, but the text dataset is
  supplied by the authors on request.
- **Sources:** [paper](https://aclanthology.org/2022.emnlp-main.677/),
  [repository and access instructions](https://github.com/blmoistawinde/EMNLP22-PsySym).

## Proposed core task suite

1. **ReDSM5 symptom classification:** predict all supported DSM-5 depression
   symptoms for a post.
2. **ReDSM5 explanation generation:** return the supporting sentence(s), graph
   evidence, and a concise rationale for each predicted symptom.
3. **PsySym symptom relevance:** retrieve and predict the relevant symptom
   class(es) for a sentence.
4. **PsySym symptom status:** distinguish an affirmed symptom from a negated,
   questioned, or uncertain mention.
5. **Optional PsySym disorder detection:** aggregate symptom evidence over a user
   history to predict disorder labels; this is downstream and should not replace
   symptom-level evaluation.

The first four tasks are the new core. They give multiple prediction and
explanation settings without requiring a new knowledge ontology for each task.
They also test the exact path the project claims to support:

```text
self-report -> semantic observation -> DSM entity/criterion -> source evidence
            -> symptom/status prediction -> grounded explanation
```

## Status of the previous IMHI tasks

| Existing task | DSM sufficiency | New role |
| --- | --- | --- |
| T-SID | Partial: depression/PTSD fit, but self-harm and control are different ontology levels | Transfer baseline after the core study |
| Dreaddit | Limited: everyday stress is not necessarily a DSM disorder | Stress-source extension |
| SAD | Low: work, finance, school, family, and other causes are contextual stressors | Psychosocial-stressor extension |
| MultiWD | Very low: wellness includes positive functioning outside a disorder taxonomy | Wellness-source extension |

These datasets should not be discarded. They demonstrate where a DSM-only graph
stops being sufficient and motivate a provenance-preserving multi-source graph.

## Additional sources required for later extensions

- **Depression symptom severity:** index the official instrument used by the
  dataset, such as PHQ-8/PHQ-9 or BDI-II, alongside DSM; BDI-Sen and DepreSym are
  suitable datasets after their data-use agreements are obtained.
- **Suicide-risk levels:** add the C-SSRS definitions and scoring semantics;
  DSM alone does not define the risk strata used by UMD Reddit Suicidality or
  C-SSRS datasets.
- **Everyday stress and stress causes:** add a psychosocial-stressor taxonomy and
  relevant contextual-condition material rather than treating stress as a DSM
  diagnosis.
- **Wellness and functioning:** add the official label definitions used by the
  dataset plus a functioning/wellness framework such as WHODAS where applicable;
  DSM remains supplementary evidence, not the label ontology.

Every external concept must retain its source and use relations such as
`related_to` or `supports`; it must not be declared `equivalent_to` a DSM concept
unless that equivalence is explicitly supported.

## Access gate

Do not train, publish, or redistribute the full ReDSM5 or PsySym text until the
respective access terms are accepted. Until access is granted, implementation
can use the public schemas and paraphrased/sample records only. No test example
may be used as a DSPy demonstration or prompt-tuning example.

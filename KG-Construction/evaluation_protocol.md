# Provenance, Difficulty, and Evaluation Protocol

<div dir="rtl" align="right" markdown="1">

این سند سه قرارداد اجباری آزمایش را مشخص می‌کند: trace قابل‌ردیابی برای هر
prediction، subsetهای سختی بدون leakage، و metricهای evidence/reasoning/cost.
همه‌ی روش‌ها باید یک artifact مشترک تولید کنند؛ مقدار `null` مجاز است، حذف فیلد
مجاز نیست.

## 1. قرارداد provenance

هر prediction یک `run_record` مستقل و versioned است:

</div>

<div dir="ltr" align="left" markdown="1">

```json
{
  "run_id": "...",
  "sample_id": "...",
  "dataset": "redsm5|psysym|...",
  "split": "pilot_dev|pilot_eval|confirmatory_test",
  "framework": "llm_only|text_rag|kg_only|hipporag2|tog|tog2",
  "bridge_enabled": true,
  "input": {"text_hash": "...", "text": "..."},
  "observations": [
    {"observation_id": "o1", "source_type": "bridge|raw_input",
     "evidence_span": "...", "char_start": 0,
     "char_end": 12, "observation": "..."}
  ],
  "retrieval": {
    "queries": [{"query_id": "q1", "source_observation_ids": ["o1"], "text": "..."}],
    "seed_entities": [{"entity_id": "...", "type": "symptom", "score": 0.0,
      "rank": 1, "source_query_ids": ["q1"]}],
    "triples": [{"triple_id": "...", "subject_id": "...", "predicate_id": "...",
      "object_id": "...", "score": 0.0, "rank": 1, "source_query_ids": ["q1"]}],
    "passages": [{"passage_id": "...", "document_id": "...", "page": null,
      "source_node_id": "...", "score": 0.0, "rank": 1,
      "source_query_ids": ["q1"]}]
  },
  "reasoning": {
    "paths": [{"path_id": "p1", "node_ids": ["..."], "edge_ids": ["..."],
      "selected_by": "ppr|beam|llm|static", "score": null}],
    "cited_evidence_ids": ["passage:...", "triple:..."],
    "stopping_reason": "answer_ready|hop_limit|no_candidate|budget_limit"
  },
  "output": {"labels": ["..."], "label_scores": {}, "explanation": "...",
    "raw_response": "...", "parse_status": "ok|repaired|failed"},
  "cost": {"latency_ms": 0, "llm_calls": 0, "input_tokens": 0,
    "output_tokens": 0, "retrieval_calls": 0, "traversal_steps": 0},
  "versions": {"code_commit": "...", "graph_snapshot": "...",
    "index_snapshot": "...", "model": "...", "prompt": "...", "config": "..."},
  "errors": []
}
```

</div>

<div dir="rtl" align="right" markdown="1">

### قواعد trace

1. `evidence_span` باید با offset به ورودی خام برگردد؛ observation بدون span فقط
   در audit به‌عنوان failure نگه‌داری می‌شود و semantic seed نمی‌سازد.
2. هر seed/triple/passage باید به query و observation مبدأ وصل باشد. در حالت
   بدون bridge، یک observation مصنوعی با نوع `raw_input` کل متن را نمایندگی می‌کند.
3. هر edge و passage باید به graph/document snapshot و provenance استخراج اولیه
   قابل resolve باشد؛ متن evidence هنگام run نیز immutable ذخیره می‌شود.
4. explanation فقط مجاز است evidenceهایی را cite کند که در همان run بازیابی
   شده‌اند. citation نامعتبر hallucinated citation محسوب می‌شود.
5. LLM-only آرایه‌های retrieval و path خالی دارد؛ نبود evidence نباید با logging
   ناقص اشتباه شود.
6. prediction نهایی، پاسخ خام مدل و parse error همگی نگه‌داری می‌شوند تا failure
   پنهان نشود. متن حساس در artifact اشتراکی با hash/ID جایگزین می‌شود.

## 2. difficulty subsets بدون leakage

سختی یک label واحد نیست؛ مجموعه‌ای از tagهای مستقل است و هر نمونه می‌تواند چند
tag داشته باشد. tagها prediction target نیستند و برای tuning استفاده نمی‌شوند.

| برچسب | تعریف عملیاتی | مثال کوتاه |
|---|---|---|
| `explicit` | نام symptom یا یک synonym بسیار نزدیک، مستقیماً در متن آمده و برای تشخیص مفهوم به استنتاج رفتاری نیاز نیست. | «I feel hopeless» برای مفهوم hopelessness. |
| `implicit` | مفهوم از توصیف تجربه یا رفتار روزمره فهمیده می‌شود، ولی نام مستقیم آن در متن نیست. یک span به‌تنهایی می‌تواند کافی باشد. | «Nothing feels enjoyable anymore» برای loss of interest. |
| `negated_or_absent` | متن صریحاً وجود symptom را رد می‌کند یا gold status آن absent/negated است؛ صرف نبودن اشاره به symptom کافی نیست. | «I am not having thoughts of hurting myself.» |
| `uncertain_or_conditional` | hedge، احتمال، سؤال، شرط یا نقل قول باعث می‌شود assertion قطعی و مستقیم نباشد. | «I might hurt myself if things get worse.» |
| `multi_symptom` | در واحد ارزیابی بیش از یک symptom gold متمایز وجود دارد، حتی اگر هرکدام در یک جملهٔ جدا باشند. | اشاره هم‌زمان به بی‌خوابی و loss of interest. |
| `shared_symptom` | symptom gold در ontology به بیش از یک disorder متصل است و به‌تنهایی disorder را یکتا نمی‌کند. این ویژگی از graph محاسبه می‌شود، نه از ظاهر متن. | sleep disturbance که در چند اختلال دیده می‌شود. |
| `temporal_or_duration` | زمان شروع، مدت، تکرار یا persistence برای تفسیر درست evidence مهم است؛ وجود هر عبارت زمانی به‌تنهایی کافی نیست. | «Nearly every day for the last three weeks.» |
| `functional_context` | متن به اثر تجربه بر کار، تحصیل، روابط، self-care یا عملکرد روزمره اشاره می‌کند و این impact بخشی از evidence قابل‌استفاده است. | «I cannot get out of bed to go to work.» |
| `compositional` | تصمیم قابل‌دفاع به ترکیب حداقل دو span یا observation مستقل نیاز دارد و هیچ‌یک به‌تنهایی evidence کامل نیست. | کاهش خواب در یک جمله و افت عملکرد در جمله‌ای دیگر. |
| `graph_multi_hop` | پس از انتخاب seed صحیح، کوتاه‌ترین مسیر معتبر در canonical graph تا target/evidence حداقل دو edge دارد. این سختی متعلق به retrieval graph است، نه لزوماً زبان متن. | behavior → symptom → criterion/passage. |
| `no_valid_seed` | در allowlist فعلی هیچ entity صحیح و قابل‌دفاعی برای شروع semantic retrieval وجود ندارد؛ سیستم باید بتواند link نکند. | موضوع روزمره‌ای که در KG پوشش ندارد. |

### توضیح و مرزبندی برچسب‌ها

#### `explicit`

این ساده‌ترین حالت linking است. معیار، حضور اصطلاحی است که انسان بدون تفسیر
رفتاری آن را به symptom وصل می‌کند. وجود کلمه‌ای هم‌ریشه اما با معنای متفاوت،
نمونه را explicit نمی‌کند. `explicit` و `implicit` برای یک target مشخص نباید
هم‌زمان فعال باشند؛ اگر نمونه چند symptom دارد، ممکن است یکی explicit و دیگری
implicit باشد و این وضعیت در annotation سطح symptom ثبت می‌شود.

#### `implicit`

متن به‌جای نام symptom، manifestation آن را توصیف می‌کند. این tag دقیقاً جایی
است که semantic bridge یا embedding باید ارزش نشان دهد. implicit بودن به معنی
multi-hop یا compositional بودن نیست: «دیگر از هیچ کاری لذت نمی‌برم» implicit
است، ولی یک span برای فهم آن کافی است.

#### `negated_or_absent`

این tag توانایی سیستم در حفظ polarity را می‌سنجد. «دربارهٔ خودکشی حرفی نزد»
negation نیست، اما «افکار خودکشی ندارم» negation است. observation، retrieval و
explanation نباید صرف بازیابی مفهوم، آن را present فرض کنند.

#### `uncertain_or_conditional`

این حالت با negation فرق دارد: assertion رد نشده، بلکه certainty یا تحقق آن
محدود است. شرط، احتمال، سؤال بلاغی، تجربهٔ فرضی و نسبت دادن گفته به فرد دیگر
در این گروه قرار می‌گیرند. سیستم باید modifier را تا prediction و explanation
حفظ کند.

#### `multi_symptom`

این tag بار استخراج و پوشش را می‌سنجد، نه ضرورت reasoning ترکیبی را. اگر متن دو
symptom مستقل و آشکار داشته باشد، `multi_symptom` است ولی لزوماً `compositional`
نیست. معیار مهم در این subset، از دست ندادن symptomهای کم‌وضوح‌تر است.

#### `shared_symptom`

این tag ابهام ontology را نشان می‌دهد. بازیابی symptom مشترک می‌تواند صحیح باشد،
ولی نتیجه‌گیری disorder از آن بدون evidence بیشتر مجاز نیست. این subset برای
سنجش اینکه traversal دانش مفید اضافه می‌کند یا صرفاً ambiguity را گسترش می‌دهد
اهمیت دارد.

#### `temporal_or_duration`

عبارت زمانی وقتی tag می‌گیرد که حذف آن interpretation را تغییر دهد؛ مثلاً شدت،
persistence یا انطباق با context یک criterion. عبارت بی‌اثر مانند «دیروز این
پست را نوشتم» به‌خودی‌خود این tag را فعال نمی‌کند. passage retrieval باید متن
اصلی را برای حفظ منطق زمان در اختیار مدل بگذارد.

#### `functional_context`

این tag وجود پیامد عملکردی را ثبت می‌کند، نه شدت بالینی یا diagnosis را. عبارت
باید یک حوزهٔ عملکرد و اختلال یا تغییر آن را نشان دهد. صرف احساس بد بدون اشاره
به عملکرد برای این tag کافی نیست.

#### `compositional`

در این حالت چند قطعه باید با هم دیده شوند. تفاوت آن با `multi_symptom` این است
که multi-symptom فقط شمار labelها را می‌گوید، اما compositional دربارهٔ ساختار
evidence است. یک نمونه می‌تواند یک label داشته باشد ولی برای اثبات آن به دو span
مکمل نیاز داشته باشد.

#### `graph_multi_hop`

این tag فقط پس از freeze شدن canonical graph و gold seed تعریف می‌شود. تعداد
جمله‌ها یا پیچیدگی زبانی ملاک نیست؛ ملاک shortest valid path است. برای جلوگیری
از مصنوعی‌کردن سختی، hubهای `code` و edgeهای نامعتبر/معکوس در محاسبهٔ مسیر مجاز
نیستند و policy گره‌های retrieval رعایت می‌شود.

#### `no_valid_seed`

این نمونه‌ها failure منبع دانش را از failure مدل جدا می‌کنند. اگر KG مفهوم مناسبی
ندارد، رفتار صحیح می‌تواند `unlinked`، text fallback یا abstention باشد. نزدیک‌ترین
entity نامرتبط نباید به‌زور به‌عنوان seed انتخاب شود.

### رابطهٔ tagها

- tagها عموماً چندبرچسبی‌اند؛ برای مثال یک نمونه می‌تواند هم `implicit`، هم
  `temporal_or_duration` و هم `graph_multi_hop` باشد.
- `explicit/implicit` در سطح هر target متقابل‌اند، ولی aggregate یک نمونهٔ
  multi-symptom می‌تواند هر دو را داشته باشد.
- `multi_symptom` دربارهٔ تعداد targetهاست؛ `compositional` دربارهٔ تعداد قطعات
  لازم برای یک تصمیم؛ و `graph_multi_hop` دربارهٔ طول مسیر در KG است.
- difficulty یک ordinal score واحد نیست. نتیجه‌ها به تفکیک tag گزارش می‌شوند و
  ساختن برچسب کلی easy/medium/hard فقط در تحلیل مکمل و با قاعدهٔ ازپیش‌ثبت‌شده
  مجاز است.

### procedure برچسب‌گذاری

1. قواعد قطعی مانند تعداد label، status و shortest-path فقط از annotation مجاز
   و snapshot ثابت محاسبه می‌شوند؛ هیچ خروجی مدل در tag دخالت ندارد.
2. `explicit/implicit`، `compositional` و اهمیت temporal/functional ابتدا توسط
   دو annotator روی pilot-development تعریف و guideline آن‌ها freeze می‌شود.
3. subset ارزیابی به‌صورت کور و بدون دیدن prediction روش‌ها annotate می‌شود؛
   اختلاف با adjudication حل و agreement گزارش می‌شود.
4. tag مربوط به graph تنها با canonical graph frozen محاسبه می‌شود. اگر target
   در گراف نیست، نمونه `no_valid_seed` است، نه شکست reasoning.
5. threshold یا تعریف tag پس از دیدن confirmatory-test تغییر نمی‌کند. تحلیل‌های
   post-hoc با همین عنوان و جدا از فرضیه‌های اصلی گزارش می‌شوند.
6. برای هر subset اندازه، توزیع label و هم‌پوشانی tagها گزارش می‌شود؛ subset کوچک
   فقط توصیفی است و مبنای ادعای قطعی قرار نمی‌گیرد.

## 3. metricهای اجباری

### 3.1 Prediction

- معیارهای اصلی: `macro-F1` و `macro-recall`؛ معیارهای سازگار با کارهای مرجع:
  `weighted-F1` و recall.
- ReDSM5 چندبرچسبی: micro/macro F1، per-label precision/recall و exact-match.
- PsySym: relevance و status جداگانه ارزیابی می‌شوند؛ disorder aggregation نتیجه‌ی
  ثانویه است.
- برای metric اصلی 95% paired bootstrap CI و اختلاف paired نسبت به baseline
  گزارش می‌شود؛ multiple comparison با Holm correction کنترل می‌شود.

### 3.2 Retrieval and linking

- `Entity Recall@k`: وجود حداقل یک canonical entity صحیح در top-k seedها.
- `MRR`: رتبه اولین seed صحیح.
- `Evidence Recall@k`: سهم gold evidence unitهایی که top-k پوشش می‌دهد.
- `Evidence Precision@k`: سهم top-k evidenceهایی که gold یا در ارزیابی انسانی
  مستقیم و مفید تشخیص داده شده‌اند.
- `Unlinked accuracy`: صحت تصمیم link نکردن روی `no_valid_seed`؛ اجبار به link
  موفقیت محسوب نمی‌شود.

### 3.3 Path validity

برای هر path چهار شرط جدا ثبت می‌شود: همه edgeها در snapshot وجود دارند؛ جهت و
predicate با graph سازگار است؛ ابتدا به seed بازیابی‌شده وصل است؛ انتها به
evidence یا concept استفاده‌شده در prediction می‌رسد. `Path validity rate` سهم
pathهایی است که هر چهار شرط را دارند. علاوه بر آن `useful path rate` با ارزیابی
کور انسانی می‌سنجد آیا path برای تصمیم مرتبط و غیرزائد بوده است. path معتبر ولی
نامرتبط، useful نیست.

### 3.4 Explanation faithfulness

- `Citation validity`: سهم citationهای resolveشدنی در evidence packet همان run.
- `Claim support`: سهم claimهای اتمی explanation که evidence cited آن‌ها را
  پشتیبانی می‌کند.
- `Contradiction rate`: سهم claimهایی که با evidence تناقض دارند.
- `Evidence coverage`: سهم labelهای پیش‌بینی‌شده که حداقل یک claim پشتیبانی‌شده
  و evidence مستقیم دارند.
- ارزیابی خودکار GraphEval برای scale استفاده می‌شود؛ subset ثابت و کور انسانی
  معیار مرجع است و agreement گزارش می‌شود. BARTScore فقط شباهت نوشتاری مکمل است.

### 3.5 Calibration and abstention

اگر score قابل‌مقایسه موجود باشد، per-label `Brier score` و `ECE` گزارش می‌شود.
thresholdها فقط روی development تعیین می‌شوند. اگر مدل score معتبر نمی‌دهد، این
metricها `not_applicable` هستند و confidence زبانی مدل جای probability را
نمی‌گیرد. برای abstention، coverage، selective accuracy/F1 و risk-coverage curve
گزارش می‌شود.

### 3.6 Efficiency and robustness

- median و p95 latency، LLM calls، input/output tokens، retrieval calls، hopها
  و estimated cost به ازای نمونه؛
- failure/empty-retrieval/parse-error rate؛
- کیفیت در برابر budget با یک سقف context و token مشترک؛
- نسبت بهبود metric اصلی به 1K input token به‌عنوان گزارش کمکی، نه معیار اصلی.

## 4. سؤال‌های پژوهشی و مقایسه‌ها

| RQ | مقایسه‌ی کنترل‌شده | معیار اصلی |
|---|---|---|
| RQ1: دانش بیرونی مفید است؟ | LLM-only در برابر text RAG/KG | prediction + faithfulness |
| RQ2: ساختار graph چه می‌افزاید؟ | text RAG در برابر KG/hybrid با budget برابر | hard-subset F1 + evidence |
| RQ3: global ranking مفید است؟ | HippoRAG2 با/بدون PPR | retrieval recall + prediction |
| RQ4: passage کنار path مفید است؟ | ToG در برابر ToG2 با traversal یکسان | groundedness + prediction |
| RQ5: bridge مفید است؟ | paired bridge/no-bridge روی همان sample و config | linking + downstream delta |
| RQ6: سود graph کجا رخ می‌دهد؟ | interaction روش × difficulty tag | paired subset delta |
| RQ7: هزینه‌ی سود چیست؟ | Pareto quality/latency/token | quality-cost frontier |

وجود شش روش به‌تنهایی ablation علّی نیست. ادعای یک مؤلفه فقط از مقایسه‌ای مجاز
است که همان مؤلفه را با بقیه تنظیمات ثابت روشن/خاموش کند.

## 5. حداقل artifactهای انتشار

- manifest شناسه‌ها و difficulty tagها بدون بازنشر متن محدودشده؛
- JSON Schema و validator برای `run_record`؛
- config و version fingerprint هر run؛
- script محاسبه metricها و paired bootstrap؛
- aggregate table و error taxonomy؛
- نمونه‌های de-identified trace برای audit، مشروط به مجوز dataset.

</div>

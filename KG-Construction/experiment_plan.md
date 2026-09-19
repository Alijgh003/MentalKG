# Final Implementation and Evaluation Plan

این سند ترتیب نهایی اجرای پروژه را مشخص می‌کند. جزئیات پل در
[`semantic_bridge.md`](semantic_bridge.md) و سیاست seedها در
[`retrieval_node_policy.md`](retrieval_node_policy.md) آمده است. انتخاب benchmark
و مرز منابع دانشی در [`dataset_selection.md`](dataset_selection.md) ثبت شده است.
قرارداد اجباری provenance، difficulty subsetها، metricها و research questionها
در [`evaluation_protocol.md`](evaluation_protocol.md) تعریف شده است.

## وضعیت فعلی پروژه — ۲۰۲۶-۰۹-۱۹

پروژه اکنون به **ابتدای فاز ۳، یعنی پیاده‌سازی شش روش retrieval/reasoning بدون
Semantic Bridge** رسیده است. زیرساخت لازم برای شروع این مقایسه آماده است:

- استخراج entity و relation از منبع و نگه‌داری provenance انجام شده است؛
- consolidation نوع‌محور entityها و predicateها انجام و snapshot آن ثبت شده است؛
- دیتابیس عملیاتی مینیمال با چهار جدول `chunks`، `entities`، `facts` و `mentions`
  ساخته و با داده‌ی کامل پر شده است؛
- فقط leaf nodeهای selected وارد `chunks` شده‌اند و chunkهای بدون mention از
  PostgreSQL و Milvus حذف شده‌اند؛ مجموعه‌ی IDهای باقی‌مانده در دو سیستم برابر
  است؛
- endpointهای relation که entity قطعی نداشتند به‌صورت entity مستقل با type برابر
  `unresolved` حفظ شده‌اند؛ اسکریپت resolve معنایی آن‌ها با شرط
  `cosine_similarity > 0.87` آماده است، اما اجرای نهایی آن یک مرحله‌ی hardening
  اختیاری پیش از freeze کردن indexهاست؛
- collectionهای entity، predicate، triple و source chunk در Milvus ساخته و از
  نظر schema، تعداد رکورد و سازگاری ID با دیتابیس جدید بررسی شده‌اند؛
- سیاست semantic seed و traversal تثبیت شده است: `symptom`، `behavior`،
  `disorder` و conceptهای منتخب برای شروع retrieval هستند و سایر typeها فقط در
  traversal و context باقی می‌مانند.

بنابراین فعالیت اصلی بعدی، پیاده‌سازی interface مشترک و سپس شش backend فاز ۳
است. پیش از pilot evaluation باید resolver اختیاری unresolvedها، بازبینی دستی
نمونه‌ای از merge/non-mergeها و smoke test نهایی retrieval اجرا و سپس snapshot
دیتابیس، collectionها و configها freeze شود.

## اصول ثابت آزمایش

- همه‌ی روش‌ها از یک مدل پاسخ‌دهنده، decoding، output schema، سقف context و
  مجموعه‌نمونه‌ی ثابت استفاده می‌کنند تا مقایسه تا حد ممکن منصفانه باشد.
- prompt، index و hyperparameterها فقط روی pilot development تنظیم و پیش از
  pilot evaluation قفل می‌شوند.
- شناسه‌ی نمونه‌ها، seed تصادفی، ورودی/خروجی مدل، evidence، زمان و token مصرفی
  برای بازتولید کامل هر run ذخیره می‌شود.
- prediction و explanation دو محور اصلی‌اند؛ efficiency محور ثانویه ولی اجباری
  گزارش است.
- همه‌ی backendها، حتی LLM-only، باید `run_record` یکسان مطابق
  `evaluation_protocol.md` تولید کنند تا نبود evidence با logging ناقص اشتباه نشود.

## فاز ۰: benchmark هسته

1. **ReDSM5:** تشخیص چندبرچسبی ۹ symptom افسردگی DSM-5 و تولید توضیح مبتنی بر
   sentence و rationale متخصص، benchmark اصلی تطابق مستقیم با گراف است.
2. **PsySym:** symptom relevance و symptom status روی ۳۸ symptom مربوط به ۷
   disorder، benchmark اصلی linking، negation/uncertainty و اشتراک symptomهاست.
3. **Optional downstream:** disorder detection در PsySym فقط بعد از موفقیت
   symptom-level اجرا می‌شود تا خطای retrieval با خطای aggregation مخلوط نشود.
4. **Secondary disease/risk-level transfer:** DR، T-SID و SWMH با validation/testهای
   ثابت‌شده در `datasets/benchmarks/dsm_grounded_v1` اجرا می‌شوند؛ این سه مجموعه
   جای benchmarkهای symptom-level اصلی را نمی‌گیرند.
5. **Previous IMHI boundary tasks:** Dreaddit، SAD و MultiWD از benchmark هسته خارج
   و به transfer/source-extension منتقل می‌شوند، چون DSM برای همه‌ی labelهایشان
   ontology کافی نیست.
6. **Access gate:** اجرای داده‌ی کامل پس از پذیرش شرایط ReDSM5 و دریافت PsySym از
   نویسندگان مجاز است؛ تا آن زمان فقط schema و sampleهای عمومی استفاده می‌شوند.

## فاز ۱: تثبیت گراف

1. **Audit و snapshot داده‌ی خام:** شمارش entity/relation/type و provenance ثبت
   می‌شود تا هیچ consolidation باعث حذف غیرقابل‌ردیابی داده نشود.
2. **Entity consolidation:** فقط entityهای مناسب، با تمرکز بر `symptom`،
   `behavior`، `disorder` و conceptهای پاک‌سازی‌شده، canonical می‌شوند تا aliasها
   به یک شناسه‌ی پایدار برسند.
3. **Predicate consolidation:** predicateهای هم‌معنا پس از entityها به vocabulary
   canonical نگاشت می‌شوند تا جهت و معنای edgeها یکدست شود.
4. **Materialize canonical graph:** tripleها با canonical entity/predicate ID
   بازسازی می‌شوند و mention، alias، source span و provenance اصلی حفظ می‌گردند.
5. **Quality gate:** نمونه‌ای از mergeها و non-mergeهای entity و predicate دستی
   ارزیابی می‌شود؛ تا precision قابل‌قبول نباشد index نهایی ساخته نمی‌شود.

## فاز ۲: ایندکس‌های embedding

embeddingهای mention و predicate خام فقط ابزار consolidation هستند؛ ایندکس‌های
retrieval نهایی بعد از تثبیت canonical graph ساخته می‌شوند:

1. **Canonical entities index:** برای semantic seeding و entity linking؛ فقط
   typeهای allowlist وارد seed index می‌شوند.
2. **Canonical predicates index:** برای relation matching و انتخاب edge در
   روش‌های تعاملی.
3. **Canonical triples index:** متن استانداردشده‌ی `(subject, predicate, object)`
   برای retrieval مستقیم fact و candidate generation ذخیره می‌شود.
4. **Source chunks index:** passageهای منبع همراه page/node/provenance برای RAG و
   کنترل منطق دقیق criterion، negation، duration و exclusion ذخیره می‌شوند.
5. **Collection-specific dimensions:** dimension جزو config هر collection است؛
   entity/predicate/triple می‌توانند ابتدا 128 و chunkها 512 یا 1024 باشند، بدون
   متغیر dimension سراسری.
6. **Retrieval smoke test:** queryهای نماینده، Recall@k، provenance و امکان
   برگشتن نتیجه‌ی خالی را می‌سنجند تا خرابی index وارد آزمایش اصلی نشود.

## فاز ۳: شش framework بدون Semantic Bridge

نسخه‌ی اولیه مستقیماً از self-report خام به‌عنوان query استفاده می‌کند:

1. **LLM-only:** بدون دانش بیرونی، برای سنجش توان پایه‌ی مدل.
2. **Vanilla text RAG:** فقط source chunk retrieval، برای اثر passage context.
3. **Static KG-only:** بازیابی subgraph/triple ثابت، برای اثر دانش ساخت‌یافته بدون
   passage.
4. **HippoRAG2-style:** entity/passage seeds و PPR، برای hybrid global ranking.
5. **ToG-style:** پیمایش تعاملی KG و ساخت reasoning path، بدون passage context.
6. **ToG2-style:** پیمایش تعاملی KG همراه passage retrieval، برای ترکیب path و
   متن منبع.

برای جلوگیری از انفجار هزینه، ابتدا یک **pilot development** کوچک برای تنظیمات
و سپس یک **pilot evaluation قفل‌شده** اجرا می‌شود. PsySym validation رسمی دارد؛
برای ReDSM5 یک development split فقط از بخش training ساخته می‌شود. test هیچ‌یک
در انتخاب prompt، روش یا hyperparameter دیده نمی‌شود. توزیع labelها حفظ و
نمونه‌های کلاس‌های نادر جداگانه در تحلیل کیفی بررسی می‌شوند.

## فاز ۴: ارزیابی baselineها و تحلیل ablation

پیش از اجرای این فاز، difficulty tagها و annotation guideline مطابق
`evaluation_protocol.md` روی development تعریف و freeze می‌شوند. هیچ tag یا
thresholdی با مشاهده‌ی خروجی confirmatory test تغییر نمی‌کند.

1. **Prediction:** weighted F1 و recall مطابق پروپوزال گزارش می‌شوند و برای آشکار
   شدن ضعف کلاس‌های اقلیت، macro-F1، recall هر کلاس، confusion matrix و bootstrap
   confidence interval نیز افزوده می‌شود.
2. **Explanation:** claim support/groundedness و ناسازگاری ادعاها با evidence از
   طریق GraphEval در اولویت‌اند؛ BARTScore فقط معیار مکمل شباهت با توضیح مرجع است.
3. **Human qualitative review:** یک نمونه‌ی ثابت از پاسخ‌های درست و غلط از نظر
   faithfulness به متن، اتکا به evidence، پوشش و خوانایی مقایسه می‌شود.
4. **Efficiency:** end-to-end latency، تعداد callهای LLM، input/output tokens،
   تعداد retrieval/traversal step و failure rate ثبت می‌شود.
5. **Ablation interpretation:** جدول capability شش framework تفاوت‌های ضمنی را
   توصیف می‌کند، اما ادعای علّی فقط برای مقایسه‌ی کنترل‌شده‌ی یک‌جزئی مانند
   HippoRAG2 با/بدون PPR و ToG2 با/بدون passage نوشته می‌شود.
6. **Trace and subset analysis:** entity/evidence retrieval، path validity،
   citation/claim faithfulness، calibration در صورت وجود score معتبر، و کیفیت
   به تفکیک difficulty tag گزارش می‌شوند.

## فاز ۵: پیاده‌سازی و کنترل کیفیت Semantic Bridge

1. extractor مینیمال مطابق `semantic_bridge_dspy.py` پیاده می‌شود تا فقط
   `evidence_span` و یک topic عمومی و retrieval-oriented را با یک گام abstraction
   معنایی از self-report خام تولید کند.
2. با بررسی غیرمتخصص‌ها، groundedness، معقول‌بودن abstraction عرفی، عدم تولید
   diagnosis، سودمندی retrieval و minimality سنجیده می‌شود؛ این مرحله هیچ gold
   label یا دانش DSM به extractor نمی‌دهد.
3. observationها برای هر backend به query مناسب تبدیل می‌شوند: متن برای RAG،
   semantic seed برای KG، و هر دو برای hybrid؛ link ضعیف مجاز است `unlinked` بماند.
4. کیفیت observation extraction از entity linking جدا گزارش می‌شود تا خطای زبان
   با خطای retrieval یا دانش بالینی مخلوط نشود.

## فاز ۶: اجرای paired با Semantic Bridge

1. همان شش framework، همان نمونه‌های pilot evaluation و همان تنظیمات قفل‌شده
   دوباره اجرا می‌شوند و تنها متغیر تغییرکرده bridge است.
2. برای هر نمونه اختلاف label، evidence، explanation، latency و tokens میان حالت
   bridge/no-bridge ذخیره می‌شود تا تحلیل paired ممکن باشد.
3. اثر bridge به تفکیک dataset، label و framework گزارش می‌شود؛ بهبود prediction
   بدون بهبود groundedness موفقیت کامل تلقی نمی‌شود.
4. failure analysis موارد حذف معنا، over-interpretation، entity link اشتباه،
   retrieval نامرتبط و reasoning ناسازگار را جداگانه کدگذاری می‌کند.

## فاز ۷: تصمیم برای اجرای نهایی

اگر pipeline و معیارها در pilot پایدار باشند، تمام configها freeze می‌شوند و شش
روش اصلی به‌همراه ablationهای ازپیش‌تعیین‌شده روی confirmatory test دست‌نخورده
اجرا می‌شوند. test رسمی ReDSM5 و PsySym پیش از freeze شدن configها استفاده
نمی‌شود. در بودجه‌ی محدود، شش روش اصلی روی confirmatory set و ablationها روی
subset قفل‌شده اجرا و محدودیت دامنه‌ی نتیجه صریحاً گزارش می‌شود.

## خروجی‌های لازم

- canonical graph و mappingهای alias/provenance؛
- چهار collection versioned همراه config و model fingerprint؛
- manifest ثابت pilot development و pilot evaluation؛
- prediction، explanation، evidence trace و cost log برای هر run؛
- difficulty manifest قفل‌شده، annotation guideline و agreement report؛
- metric scripts برای retrieval، path validity، faithfulness، calibration و
  paired confidence interval؛
- جدول اصلی شش روش، جدول paired bridge/no-bridge و گزارش failure analysis.

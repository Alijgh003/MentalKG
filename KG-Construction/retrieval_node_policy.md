# Retrieval and reasoning node policy

## Decision

The primary query is a patient's natural-language self-report, not a request
about a DSM criterion label or an ICD code. Initial semantic retrieval must use
an explicit allowlist of clinically meaningful seed types:

```text
symptom
behavior
disorder
selected, cleaned concept mentions
```

Do **not** include `code`, `criterion`, or `patient` mentions in the semantic
seed index. Here, `patient` normally denotes a named person in a DSM/textbook
case example; it is neither the current patient issuing the self-report nor a
reusable clinical concept. Do not spend consolidation or standalone-embedding
effort on these types for the current use case. They remain in the knowledge
graph as context-dependent intermediate nodes and retain their source node,
hierarchy, and raw payload.

`duration` and `specifier` are candidates for later inclusion only after
retrieval evaluation demonstrates value. Prefer an allowlist so a newly added
entity type does not become seed-eligible by default.

## Why

- Values such as `G31.84` are identifiers, not useful semantic representations.
  The same code can connect several different contextual diagnoses and become a
  high-degree hub. Equal code strings do not imply equal clinical assertions.
- Values such as `B`, `A3`, `Criterion E5`, or `Criteria A-E` have no stable
  meaning without their owning disorder/episode and document hierarchy.
- Named `patient` nodes mostly identify case-example subjects. Seeding from
  their names would retrieve a particular narrative rather than semantically
  match the current patient's self-report. They remain useful during traversal
  for reaching the example's symptoms, behaviors, diagnoses, and evidence.
- A database sample found direct relation links for 3,387 of 3,477 short or
  ambiguous criterion mentions (97.4%) and 2,115 of 2,142 descriptive criterion
  mentions (98.7%). Relations plus source context therefore provide adequate
  access without global criterion consolidation.
- Existing relations can be incomplete or inconsistently directed. They are
  useful navigation evidence, not a replacement for the original criterion
  text or a safe basis for automated diagnosis.

## Expected retrieval flow

```text
patient self-report
  -> semantic seed retrieval (symptoms, behaviors, disorders, concepts)
  -> graph traversal
  -> criteria, durations, specifiers, exclusions, and other intermediate nodes
  -> supporting source passages
  -> evidence-grounded response
```

This matches graph-retrieval designs in which semantic matches initialize a
walk/search, while other nodes remain available for multi-hop expansion. In a
Personalized PageRank-style retriever, excluded types receive no restart/seed
mass but may remain in the transition graph. In a beam-search-style graph
reasoner, they are not initial candidates but may be selected at later hops.

## Node capabilities

Keep these capabilities separate rather than treating every node as equally
retrievable:

| Entity type | Semantic seed | Traversable | Typical answer/evidence role |
| --- | --- | --- | --- |
| `symptom` | yes | yes | clinical concept |
| `behavior` | yes | yes | clinical concept |
| `disorder` | yes | yes | clinical hypothesis/concept |
| selected `concept` | yes | yes | clinical concept |
| `duration` | not initially | yes | constraint |
| `specifier` | not initially | yes | qualifier |
| `patient` | no | yes, within case examples | case/example subject |
| `criterion` | no | yes | contextual constraint/reference |
| `code` | no | yes, preferably low weight | identifier/bridge |
| `section_title` | no | yes | structural context |
| `example_label` | no | yes | structural/example context |

Suggested implementation properties are `seed_eligible`,
`traversal_eligible`, `answer_eligible`, and optionally `traversal_weight`.
Seed eligibility must not be inferred from traversal eligibility.

## Guardrails

1. Keep criterion identity local (for example, extraction set + node ID +
   ordinal). Never globally merge labels such as `B` or `A1`.
2. Normalize an ICD code string only as an identifier. Do not merge diagnoses
   merely because they share a code.
3. Preserve source text and hierarchy for every contextual node. Retrieve the
   passage when exact criterion logic, negation, cardinality, duration, or
   exclusion matters.
4. Penalize or gate traversal through high-degree code nodes so shared billing
   codes do not leak retrieval into unrelated etiologies.
5. Prioritize relation normalization (predicate vocabulary and edge direction)
   over `code`/`criterion` entity consolidation.
6. Validate this policy with retrieval metrics on representative patient
   self-report questions before expanding the seed allowlist.
7. Do not merge the current user/patient with named `patient` nodes from book
   examples. Those nodes are provenance-scoped narrative subjects only.

---

# نسخه فارسی: سیاست گره‌ها در بازیابی و استدلال

## تصمیم

کوئری اصلی پروژه، خوداظهاری بیمار به زبان طبیعی است؛ نه پرسش مستقیم درباره‌ی
یک برچسب criterion یا یک کد ICD. بازیابی معنایی اولیه باید از allowlist صریح
زیر استفاده کند:

```text
symptom
behavior
disorder
mentionهای منتخب و پاک‌سازی‌شده‌ی concept
```

mentionهای `code`، `criterion` و `patient` را وارد semantic seed index نکنید.
در این گراف، `patient` معمولاً نام فردی در case example کتاب است؛ نه بیمار فعلی
که خوداظهاری را مطرح کرده و نه یک مفهوم بالینی قابل‌استفاده‌ی مجدد. در کاربرد
فعلی برای consolidation یا embedding مستقل این typeها هزینه نکنید. این
mentionها به‌صورت گره‌های میانی و وابسته به context در گراف باقی می‌مانند و
باید source node، hierarchy و raw payload آن‌ها حفظ شود.

`duration` و `specifier` فقط در صورتی بعداً به seedها اضافه شوند که ارزیابی
retrieval مفید بودنشان را نشان دهد. از allowlist استفاده کنید تا typeهای جدید
به‌طور پیش‌فرض seed-eligible نشوند.

## دلیل

- مقداری مانند `G31.84` یک شناسه است، نه یک بازنمایی معنایی مناسب. یک کد ممکن
  است چند diagnosis وابسته به context را به هم وصل کند و به hub پرتراکم تبدیل
  شود. یکسان بودن رشته‌ی کد به معنی یکسان بودن clinical assertion نیست.
- مقادیری مانند `B`، `A3`، `Criterion E5` یا `Criteria A-E` بدون disorder یا
  episode مالک و hierarchy سند معنای پایداری ندارند.
- nodeهای `patient` عمدتاً افراد نام‌گذاری‌شده در مثال‌های بالینی کتاب‌اند.
  seed شدن نام آن‌ها یک روایت خاص را بازیابی می‌کند، نه اینکه خوداظهاری بیمار
  فعلی را از نظر معنایی match کند. با این حال در traversal برای رسیدن به symptom،
  behavior، diagnosis و evidence همان مثال مفیدند.
- نمونه‌گیری از دیتابیس نشان داد ۳٬۳۸۷ مورد از ۳٬۴۷۷ criterion کوتاه یا مبهم
  (۹۷٫۴٪) و ۲٬۱۱۵ مورد از ۲٬۱۴۲ criterion توصیفی (۹۸٫۷٪) relation مستقیم
  دارند. بنابراین relation و source context بدون consolidation سراسری دسترسی
  کافی فراهم می‌کنند.
- relationهای فعلی ممکن است ناقص، دارای predicate ناهماهنگ یا با جهت معکوس
  باشند. آن‌ها evidence خوبی برای navigation هستند، اما جای متن اصلی criterion
  را نمی‌گیرند و مبنای امنی برای تشخیص خودکار نیستند.

## جریان مورد انتظار بازیابی

```text
خوداظهاری بیمار
  -> بازیابی semantic seed از symptom، behavior، disorder و concept
  -> پیمایش گراف
  -> criterion، duration، specifier، exclusion و دیگر گره‌های میانی
  -> passageهای منبع
  -> پاسخ مبتنی بر evidence
```

در بازیابی مبتنی بر Personalized PageRank، typeهای حذف‌شده restart/seed mass
نمی‌گیرند، اما در transition graph باقی می‌مانند. در reasoner مبتنی بر beam
search نیز candidate اولیه نیستند، ولی در hopهای بعدی قابل انتخاب‌اند.

## قابلیت هر نوع گره

| نوع انتیتی | Semantic seed | قابل پیمایش | نقش معمول |
| --- | --- | --- | --- |
| `symptom` | بله | بله | مفهوم بالینی |
| `behavior` | بله | بله | مفهوم بالینی |
| `disorder` | بله | بله | مفهوم یا فرضیه بالینی |
| `concept` منتخب | بله | بله | مفهوم بالینی |
| `duration` | فعلاً خیر | بله | محدودیت زمانی |
| `specifier` | فعلاً خیر | بله | qualifier |
| `patient` | خیر | بله، در محدوده case example | فرد موضوع مثال یا روایت |
| `criterion` | خیر | بله | constraint یا reference وابسته به context |
| `code` | خیر | بله، ترجیحاً با وزن کم | identifier یا bridge |
| `section_title` | خیر | بله | context ساختاری |
| `example_label` | خیر | بله | context ساختاری یا مثال |

ویژگی‌های پیشنهادی برای پیاده‌سازی عبارت‌اند از `seed_eligible`،
`traversal_eligible`، `answer_eligible` و در صورت نیاز `traversal_weight`.
قابلیت seed شدن نباید از قابلیت پیمایش استنتاج شود.

## قواعد محافظتی

1. identity هر criterion را محلی نگه دارید؛ مثلاً ترکیب extraction set، node ID
   و ordinal. برچسب‌هایی مانند `B` یا `A1` را سراسری merge نکنید.
2. رشته‌ی ICD را فقط به‌عنوان identifier نرمال کنید. diagnosisهایی را که یک
   کد مشترک دارند با هم merge نکنید.
3. متن منبع و hierarchy را برای تمام گره‌های وابسته به context حفظ کنید. هرجا
   منطق دقیق criterion، نفی، cardinality، duration یا exclusion مهم است، passage
   اصلی را بازیابی کنید.
4. پیمایش از code nodeهای پرتراکم را محدود یا جریمه کنید تا کد مشترک retrieval
   را به etiologyهای نامرتبط نشت ندهد.
5. نرمال‌سازی vocabulary و جهت relationها را بر consolidation نوع‌های `code`
   و `criterion` مقدم بدانید.
6. پیش از گسترش seed allowlist، این سیاست را با معیارهای retrieval روی مجموعه‌ای
   نماینده از پرسش‌های مبتنی بر خوداظهاری بیمار ارزیابی کنید.
7. بیمار فعلی یا کاربر را با nodeهای `patient` مربوط به مثال‌های کتاب merge
   نکنید. این nodeها فقط subjectهای روایی و وابسته به provenance هستند.

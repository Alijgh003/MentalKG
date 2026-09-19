# یادداشت کوتاه Entity Consolidation

هدف این مرحله کم‌کردن تکرارهای لفظی در entityها و predicateها بود، بدون حذف
mention خام یا provenance آن. ورودی، clusterهای type-specific ساخته‌شده از
embeddingهای Milvus بود. فقط `symptom`، `behavior`، `disorder` و `concept` در
entity consolidation شرکت کردند؛ `code`، `criterion`، `patient`، `duration`،
`specifier` و گره‌های ساختاری برای traversal حفظ شدند ولی canonical semantic
entity نگرفتند.

درون هر cluster ابتدا متن‌های دقیقاً یکسان با normalization برابر
`strip + casefold` یکی شدند. سپس به‌صورت iterative نماینده‌ای انتخاب شد که بیشترین
mention را مستقیماً پوشش می‌داد. هر عضو فقط وقتی به آن نماینده وصل شد که cosine
similarity مستقیم آن‌ها **بزرگ‌تر از 0.87** بود. از transitive connected components
استفاده نشد تا شباهت `A↔B` و `B↔C` باعث merge شدن ناموجه `A↔C` نشود. تعداد
clusterهای اولیه وابسته به type و اندازه vocabulary بود؛ آن clusterها فقط candidate
bucket بودند، نه کلاس بالینی یا تصمیم نهایی merge.

## Snapshot قبل و بعد

| گروه | mention خام | متن یکتا | canonical | exact match | cosine match |
|---|---:|---:|---:|---:|---:|
| symptom | 25,502 | 11,352 | 7,167 | 7,125 | 11,210 |
| behavior | 1,164 | 919 | 759 | 214 | 191 |
| disorder | 21,283 | 3,543 | 1,708 | 8,662 | 10,913 |
| concept | 22,673 | 10,223 | 7,016 | 7,449 | 8,208 |
| predicate | 76,442 | 6,705 | 4,296 | 40,856 | 31,290 |

پوشش mapping برای چهار entity type و predicateها ۱۰۰٪ شد. کمترین cosine ثبت‌شده
در mergeها کمی بیشتر از 0.87 است. اجرای ثبت‌شده با شناسه
`ddcc45f1-6275-402e-abde-da839a10b39b`، الگوریتم
`direct-representative-v1` و threshold برابر `0.87` انجام شد.

## آنچه حفظ شد

- همه‌ی `entity_mentions` و `relation_mentions` خام، source path، node، ordinal و
  payload آن‌ها باقی ماند؛
- هر mention پردازش‌شده به `canonical_entity_id` یا `canonical_predicate_id` وصل شد؛
- نوع تطبیق (`representative`، `exact` یا `cosine`) و similarity مستقیم ذخیره شد؛
- canonical row نام نماینده، representative mention، member count، cluster و run
  تولیدکننده را نگه می‌دارد؛
- manifest hash و summary هر اجرا در `entity_consolidation_runs` ثبت شد تا نتیجه
  قابل بازتولید و audit باشد.

این طراحی canonical identity را از provenance جدا نگه می‌دارد: retrieval و graph
می‌توانند روی canonicalها کار کنند، ولی هر نتیجه همچنان به mention و متن منبع اصلی
برمی‌گردد.

# Agent Notes

This repository builds a DSM-derived knowledge graph for semantic retrieval and
multi-hop reasoning over patient self-reports. Before changing entity
consolidation, embeddings, or graph retrieval, read
[`KG-Construction/retrieval_node_policy.md`](KG-Construction/retrieval_node_policy.md).

Key invariant: `code`, `criterion`, and `patient` mentions are graph context,
not semantic seed entities. `patient` normally identifies a named person in a
book case example, not the current user/patient or a reusable clinical concept.
Preserve these mentions and their provenance for traversal, but exclude them
from the initial semantic entity index. Concentrate consolidation quality on
clinically meaningful seed types such as `symptom`, `behavior`, `disorder`, and
selected `concept` mentions.

Other work may be in progress in this checkout. Preserve unrelated changes and
do not clean or revert a dirty worktree.

## نسخه فارسی

این مخزن برای ساخت گراف دانش مبتنی بر DSM با هدف بازیابی معنایی و استدلال
چندمرحله‌ای روی خوداظهاری بیمار است. پیش از تغییر در consolidation انتیتی‌ها،
embeddingها یا بازیابی گراف، فایل
[`KG-Construction/retrieval_node_policy.md`](KG-Construction/retrieval_node_policy.md)
را بخوانید.

قاعده اصلی: mentionهای دارای type برابر با `code`، `criterion` و `patient`
انتیتی اولیه برای جستجوی معنایی نیستند. `patient` معمولاً نام یک فرد در case
exampleهای کتاب است، نه بیمار فعلی یا یک مفهوم بالینی قابل‌استفاده‌ی مجدد.
این mentionها و provenance آن‌ها را به‌عنوان context و گره میانی برای پیمایش
گراف حفظ کنید، اما وارد semantic seed index نکنید. تمرکز consolidation باید
روی typeهای بالینی و مناسب seed مانند `symptom`، `behavior`، `disorder` و
mentionهای منتخب و پاک‌سازی‌شده‌ی `concept` باشد.

ممکن است افراد یا ایجنت‌های دیگری هم‌زمان روی همین checkout کار کنند. تغییرات
نامرتبط را حفظ کنید و ورک‌تری کثیف را پاک یا revert نکنید.

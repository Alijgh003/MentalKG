# Minimal Semantic Bridge

## تعریف نهایی

**پل معنایی** یک بازنمایی میانی، مینیمال و قابل‌ردیابی میان زبان روزمره‌ی
خوداظهاری و ورودی سیستم‌های retrieval است. این پل spanهای معنادار متن را با
**یک گام انتزاع معنایی عرفی** به topicهای عمومی و قابل‌بازیابی تبدیل می‌کند؛
برای مثال «ساعت‌ها گریه کردن» می‌تواند به «اندوه شدید یا پریشانی عاطفی» و
«نوشیدن تا blackout» به «مصرف زیاد الکل» تبدیل شود. این تبدیل از paraphrase
صرف فراتر می‌رود، اما وارد تشخیص روان‌پزشکی، معیار DSM یا برچسب task نمی‌شود.
هر observation یک عبارت شاهد عینی و یک topic روزمره دارد که یک فرد غیرمتخصص
بتواند ارتباط میان آن دو را با دانش عمومی تأیید کند.

DSM و گراف **بعد از پل** وارد می‌شوند: گراف مسیر رسیدن از observation به
مفاهیم، روابط و passageهای منبع را فراهم می‌کند و DSM شواهد و ساختار بالینی
قابل‌استناد را در اختیار روش retrieval/reasoning می‌گذارد. match شدن با یک
انتیتی DSM صرفاً یک candidate retrieval است و به‌تنهایی تشخیص یا label نیست.

## قرارداد حداقلی خروجی

برای هر observation فقط دو فیلد تولید می‌شود:

```json
{
  "evidence_span": "verbatim span from the self-report",
  "observation": "normalized general-language retrieval topic"
}
```

نفی، تردید، شرط، زمان و شخص تجربه‌کننده نباید به aspectهای جدا تبدیل شوند؛ اگر
معنای topic را تغییر می‌دهند، باید در خود `observation` حفظ شوند. یک span
می‌تواند چند topic واقعاً متمایز تولید کند، اما topicهای تکراری یا جزئیات
کم‌فایده نباید جدا شوند. خروجی برای متن فاقد خوداظهاری substantive می‌تواند خالی
باشد.

### مرز انتزاع

```text
بیش از حد خام:     The writer says they cried for hours.
سطح مطلوب bridge: Prolonged crying associated with intense emotional distress.
بیش از حد تخصصی:  A depressive episode or a DSM depressive symptom.
```

پل اجازه دارد synonymها را normalize کند، احساس یا رفتار آشکار را به topic عرفی
تبدیل کند و استعاره‌ی واضح را با توجه به context بفهمد. پل اجازه ندارد diagnosis،
DSM criterion، label دیتاست، علت بیان‌نشده، duration بیان‌نشده یا شدت بیشتر از
آنچه span پشتیبانی می‌کند اضافه کند.

## procedure

```text
raw self-report
  -> one-shot minimal observation extraction
  -> audit: every observation has a verbatim supporting span
  -> use each normalized observation as a retrieval query
  -> retrieve top-k candidates from the appropriate index
  -> threshold/rerank; keep "unlinked" when evidence is weak
  -> for KG methods, traverse from linked seed entities
  -> retrieve relations, constraints, and source passages
  -> build a provenance-preserving evidence packet
  -> task model predicts a label and writes an evidence-grounded explanation
```

استخراج‌کننده فقط `self_report` را می‌بیند؛ سؤال task، مجموعه‌ی labelها، پاسخ
مرجع، DSM و نتایج retrieval نباید به آن داده شوند. این جداسازی مانع تبدیل پل
به یک classifier پنهان یا استخراج task-specific می‌شود. topicهای خروجی باید
برای retrieval عمومی مفید باشند، نه اینکه پاسخ یک task خاص را encode کنند.

### استفاده در شش روش

- در **LLM-only** فقط observationهای زبانی به مدل پاسخ‌دهنده داده می‌شوند و هیچ
  دانش بیرونی اضافه نمی‌شود.
- در **Vanilla RAG** observationها query بازیابی passage هستند.
- در **Static KG-only** observationها به semantic seedهای گراف link می‌شوند.
- در **HippoRAG2** observationها seedهای phrase/entity و passage را آغاز می‌کنند.
- در **ToG** observationها نقطه‌ی شروع پیمایش تعاملی گراف هستند.
- در **ToG2** observationها هم graph traversal و هم passage retrieval را آغاز
  می‌کنند.

در نتیجه نسخه‌ی bridge-assisted روش LLM-only همچنان LLM-only است، چون فقط یک
بازنمایی زبانی از همان ورودی دریافت می‌کند و هیچ شاهد DSM یا گرافی به آن داده
نمی‌شود.

## مثال کامل

ورودی واقعی از T-SID:

```text
I'm going to kill myself if my job gets any busier.
```

خروجی مجاز پل:

```json
[
  {
    "evidence_span": "I'm going to kill myself if my job gets any busier",
    "observation": "A conditional statement about killing oneself if work becomes busier."
  }
]
```

در observation شرط جمله حفظ شده است، اما پل عبارت‌هایی مانند `suicidal ideation`،
`depression`، `occupational stress disorder` یا label دیتاست را تولید نکرده
است. مرحله‌ی بعد می‌تواند observation را embed کند و candidateهایی مانند مفهوم
مرتبط با self-harm/suicide statement یا passageهای مرتبط را برگرداند. سپس گراف
ممکن است به روابط و متن DSM برسد، ولی تصمیم درباره‌ی label
`suicide_or_self_harm` فقط در task model و با evidence packet انجام می‌شود.

نمونه‌ای که باید از over-interpretation جلوگیری کند:

```text
Besides drinking and crying until I pass out.
```

```json
[
  {
    "evidence_span": "drinking and crying until I pass out",
    "observation": "Heavy alcohol use to the point of passing out."
  },
  {
    "evidence_span": "drinking and crying until I pass out",
    "observation": "Intense crying associated with severe emotional distress."
  }
]
```

این دو topic از بازگویی لفظی فراتر می‌روند و برای retrieval مفیدند، اما پل نباید
از این جمله به‌تنهایی علت رفتار، اختلال مصرف الکل، افسردگی یا وضعیت social
wellness را نتیجه بگیرد. عبارت `pass out` نیز نباید به `death` تبدیل شود.

## معیار پذیرش خود پل

یک نمونه‌ی ثابت و متوازن از taskها را افراد غیرمتخصص بررسی می‌کنند و برای هر
observation پنج سؤال باینری پاسخ می‌دهند: آیا span عیناً در متن است؟ آیا topic با
یک استنتاج عرفی قابل‌قبول از span به دست می‌آید؟ آیا diagnosis یا جزئیات
پشتیبانی‌نشده اضافه نشده است؟ آیا topic برای retrieval مفید است؟ آیا observation
غیرتکراری و لازم است؟ خطای هرکدام failure محسوب می‌شود. کیفیت entity linking
جدا از کیفیت پل و با Recall@k/MRR و امکان `unlinked` ارزیابی می‌شود.

## محدودیت‌های retrieval

برای semantic entity linking فقط `symptom`، `behavior`، `disorder` و
`concept`های منتخب و پاک‌سازی‌شده seed-eligible هستند. گره‌های `code`،
`criterion` و `patient` seed نیستند، اما پس از linking می‌توانند برای traversal
و provenance استفاده شوند. فرد جاری هرگز با `patient`های case example ادغام
نمی‌شود.

DSPy pseudocode این قرارداد در
[`semantic_bridge_dspy.py`](semantic_bridge_dspy.py) نگه‌داری می‌شود.

## سیاست few-shot

prompt اجرایی از مثال‌های synthetic استفاده می‌کند، نه نمونه‌های test. مجموعه‌ی
demoها عمداً مرز یک‌گام abstraction را پوشش می‌دهد: تبدیل رفتار به topic عمومی،
زمان، نفی، نسبت دادن گفته به شخص دیگر، عدم قطعیت، شرط، استعاره و یک پرسش
غیرخوداظهاری با خروجی خالی. مثال‌ها label، مفهوم DSM یا تفسیر علت را آموزش
نمی‌دهند؛ تبدیل grounded و retrieval-oriented از
`evidence_span -> observation` را نشان می‌دهند.

نمونه‌های واقعی دیتاست که در بخش مثال کامل این سند آمده‌اند صرفاً برای توضیح
طراحی هستند و نباید به demoهای inference روی همان benchmark تبدیل شوند. در صورت
استفاده از optimizerهای DSPy نیز candidate demoها باید فقط از داده‌ی synthetic
یا train/development مجاز انتخاب شوند و evaluation/test همواره held-out بماند.

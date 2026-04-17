"""
Urdu-English Parallel Dataset Generator
Covers domains: greetings, daily life, nature, emotions, food, time, family
"""

import pandas as pd
import numpy as np
import re
import unicodedata

# ─── Raw bilingual pairs ────────────────────────────────────────────────────

RAW_PAIRS = [
    # Greetings & social
    ("آپ کیسے ہیں؟", "How are you?"),
    ("میں ٹھیک ہوں۔", "I am fine."),
    ("آپ کا نام کیا ہے؟", "What is your name?"),
    ("میرا نام احمد ہے۔", "My name is Ahmad."),
    ("خوش آمدید!", "Welcome!"),
    ("شکریہ۔", "Thank you."),
    ("معاف کریں۔", "Excuse me."),
    ("براہ کرم۔", "Please."),
    ("خدا حافظ۔", "Goodbye."),
    ("صبح بخیر۔", "Good morning."),
    ("شام بخیر۔", "Good evening."),
    ("رات بخیر۔", "Good night."),
    ("ملتے ہیں۔", "See you later."),
    ("آپ سے مل کر خوشی ہوئی۔", "Nice to meet you."),
    ("آپ کہاں سے ہیں؟", "Where are you from?"),
    ("میں پاکستان سے ہوں۔", "I am from Pakistan."),
    ("آپ کیا کرتے ہیں؟", "What do you do?"),
    ("میں طالب علم ہوں۔", "I am a student."),
    ("میں استاد ہوں۔", "I am a teacher."),
    ("میں ڈاکٹر ہوں۔", "I am a doctor."),
    # Daily life
    ("میں کھانا کھا رہا ہوں۔", "I am eating food."),
    ("پانی پینا صحت کے لیے اچھا ہے۔", "Drinking water is good for health."),
    ("میں سو رہا ہوں۔", "I am sleeping."),
    ("وہ کتاب پڑھ رہا ہے۔", "He is reading a book."),
    ("بچے کھیل رہے ہیں۔", "The children are playing."),
    ("وہ گھر جا رہی ہے۔", "She is going home."),
    ("میں بازار جاتا ہوں۔", "I go to the market."),
    ("آج موسم اچھا ہے۔", "The weather is nice today."),
    ("بارش ہو رہی ہے۔", "It is raining."),
    ("آج گرمی ہے۔", "It is hot today."),
    ("آج سردی ہے۔", "It is cold today."),
    ("سورج نکل آیا ہے۔", "The sun has risen."),
    ("رات ہو گئی ہے۔", "It has become night."),
    ("میں کام کر رہا ہوں۔", "I am working."),
    ("وہ گانا گا رہی ہے۔", "She is singing a song."),
    ("ہم سفر کر رہے ہیں۔", "We are traveling."),
    ("میں نے خط لکھا۔", "I wrote a letter."),
    ("دروازہ بند کرو۔", "Close the door."),
    ("کھڑکی کھولو۔", "Open the window."),
    ("بتی جلاؤ۔", "Turn on the light."),
    # Family
    ("میری ماں بہت اچھی ہیں۔", "My mother is very kind."),
    ("میرے والد محنتی ہیں۔", "My father is hardworking."),
    ("میری بہن ڈاکٹر ہے۔", "My sister is a doctor."),
    ("میرا بھائی انجینئر ہے۔", "My brother is an engineer."),
    ("دادا جان بیمار ہیں۔", "Grandfather is sick."),
    ("نانی جان کہانیاں سناتی ہیں۔", "Grandmother tells stories."),
    ("بچہ رو رہا ہے۔", "The child is crying."),
    ("خاندان اکٹھا ہے۔", "The family is together."),
    ("میری شادی ہو گئی ہے۔", "I am married."),
    ("ہمارے تین بچے ہیں۔", "We have three children."),
    # Food
    ("مجھے بریانی پسند ہے۔", "I like biryani."),
    ("چائے گرم ہے۔", "The tea is hot."),
    ("روٹی تازہ ہے۔", "The bread is fresh."),
    ("دودھ صحت کے لیے مفید ہے۔", "Milk is beneficial for health."),
    ("پھل کھانا فائدہ مند ہے۔", "Eating fruit is beneficial."),
    ("سبزیاں کھاؤ۔", "Eat vegetables."),
    ("کھانا تیار ہے۔", "The food is ready."),
    ("مجھے بھوک لگی ہے۔", "I am hungry."),
    ("پیاس لگی ہے۔", "I am thirsty."),
    ("یہ مزیدار ہے۔", "This is delicious."),
    # Emotions
    ("میں خوش ہوں۔", "I am happy."),
    ("وہ اداس ہے۔", "He is sad."),
    ("مجھے غصہ آ رہا ہے۔", "I am getting angry."),
    ("وہ ڈری ہوئی ہے۔", "She is scared."),
    ("مجھے تھکاوٹ ہے۔", "I am tired."),
    ("وہ بہت خوش ہے۔", "He is very happy."),
    ("مجھے پریشانی ہے۔", "I am worried."),
    ("آپ کا شکریہ، بہت مہربانی۔", "Thank you, very kind of you."),
    ("مجھے یقین ہے۔", "I am confident."),
    ("وہ حیران ہے۔", "He is surprised."),
    # Nature
    ("درخت بڑے ہیں۔", "The trees are tall."),
    ("پہاڑ بہت اونچا ہے۔", "The mountain is very high."),
    ("دریا بہہ رہا ہے۔", "The river is flowing."),
    ("پھول کھل رہے ہیں۔", "The flowers are blooming."),
    ("پرندے گا رہے ہیں۔", "The birds are singing."),
    ("آسمان نیلا ہے۔", "The sky is blue."),
    ("رات کو ستارے چمکتے ہیں۔", "Stars shine at night."),
    ("چاند خوبصورت ہے۔", "The moon is beautiful."),
    ("ہوا چل رہی ہے۔", "The wind is blowing."),
    ("برف پڑ رہی ہے۔", "It is snowing."),
    # Time
    ("ابھی کیا وقت ہے؟", "What time is it now?"),
    ("صبح کے آٹھ بجے ہیں۔", "It is eight in the morning."),
    ("آج پیر کا دن ہے۔", "Today is Monday."),
    ("کل جمعہ ہے۔", "Tomorrow is Friday."),
    ("پچھلے سال میں گیا تھا۔", "I went last year."),
    ("اگلے مہینے امتحان ہے۔", "The exam is next month."),
    ("وقت بہت قیمتی ہے۔", "Time is very valuable."),
    ("جلدی کرو۔", "Hurry up."),
    ("دیر ہو گئی۔", "It is late."),
    ("ابھی آتا ہوں۔", "I am coming right now."),
    # Education
    ("یہ سوال مشکل ہے۔", "This question is difficult."),
    ("میں نے امتحان دیا۔", "I took the exam."),
    ("وہ اسکول جاتا ہے۔", "He goes to school."),
    ("کتاب پڑھنا ضروری ہے۔", "Reading books is necessary."),
    ("استاد نے سمجھایا۔", "The teacher explained."),
    ("مجھے ریاضی پسند ہے۔", "I like mathematics."),
    ("وہ بہت ذہین ہے۔", "He is very intelligent."),
    ("محنت کامیابی کی چابی ہے۔", "Hard work is the key to success."),
    # Places
    ("لاہور ایک خوبصورت شہر ہے۔", "Lahore is a beautiful city."),
    ("کراچی پاکستان کا سب سے بڑا شہر ہے۔", "Karachi is the largest city of Pakistan."),
    ("مسجد میں نماز پڑھو۔", "Pray in the mosque."),
    ("ہسپتال قریب ہے۔", "The hospital is nearby."),
    ("یہ راستہ لمبا ہے۔", "This road is long."),
    ("بازار شور شرابا ہے۔", "The market is noisy."),
    # Health
    ("میرا سر درد ہو رہا ہے۔", "I have a headache."),
    ("ڈاکٹر کے پاس جاؤ۔", "Go to the doctor."),
    ("دوائی وقت پر کھاؤ۔", "Take medicine on time."),
    ("صحت سب سے بڑی نعمت ہے۔", "Health is the greatest blessing."),
    ("ورزش روزانہ کرو۔", "Exercise daily."),
]


# ─── Feature Engineering ─────────────────────────────────────────────────────

def compute_features(pairs):
    rows = []
    for ur, en in pairs:
        ur_tokens = ur.split()
        en_tokens = en.split()
        rows.append({
            "urdu": ur,
            "english": en,
            # token counts
            "urdu_len": len(ur_tokens),
            "english_len": len(en_tokens),
            # char counts
            "urdu_char_len": len(ur),
            "english_char_len": len(en),
            # length ratio (en/ur tokens)
            "length_ratio": round(len(en_tokens) / max(len(ur_tokens), 1), 3),
            # ends with question mark?
            "is_question": int(ur.strip().endswith("؟") or en.strip().endswith("?")),
            # ends with exclamation?
            "is_exclamation": int(ur.strip().endswith("!") or en.strip().endswith("!")),
            # average word length urdu
            "avg_word_len_urdu": round(np.mean([len(w) for w in ur_tokens]), 2),
            # average word length english
            "avg_word_len_english": round(np.mean([len(w) for w in en_tokens]), 2),
            # unique urdu tokens
            "unique_urdu_tokens": len(set(ur_tokens)),
            # unique english tokens
            "unique_english_tokens": len(set(en_tokens)),
        })
    return pd.DataFrame(rows)


def clean_text(text, lang="ur"):
    """Normalise text for tokenisation."""
    text = text.strip()
    if lang == "ur":
        # Remove diacritics (harakat) but keep Urdu letters & punctuation
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', text)
        # Normalise Urdu punctuation
        text = text.replace('۔', ' ۔').replace('؟', ' ؟').replace('!', ' !')
    else:
        text = text.lower()
        text = re.sub(r"[^a-z0-9 '.,!?]", '', text)
        text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_dataset():
    df = compute_features(RAW_PAIRS)
    df["urdu_clean"] = df["urdu"].apply(lambda x: clean_text(x, "ur"))
    df["english_clean"] = df["english"].apply(lambda x: clean_text(x, "en"))
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("urdu_english_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"Dataset saved: {len(df)} pairs")
    print(df[["urdu", "english", "urdu_len", "english_len", "is_question"]].head(10).to_string())

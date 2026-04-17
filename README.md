# 🌐 Urdu ↔ English Neural Machine Translation

A sequence-to-sequence neural machine translation system built entirely in **Keras / TensorFlow**.

---

## Architecture

```
INPUT (Urdu tokens)
       │
       ▼
┌──────────────────────────────┐
│   Embedding Layer (dim=128)  │
└──────────────┬───────────────┘
               │
       ┌───────▼────────┐
       │  BiLSTM Encoder│   ← returns sequences + merged fwd/bwd states
       │  (256 units×2) │
       └───────┬────────┘
               │ enc_out (batch, src_len, 512)
               │ state_h, state_c  (batch, 512)
               │
       ┌───────▼───────────────────────┐
       │  Bahdanau Attention           │
       │  score = V(tanh(W1·enc +      │
       │               W2·dec_h))      │
       │  context = Σ(softmax·enc_out) │
       └───────┬───────────────────────┘
               │ context (batch, 512)
               │
       ┌───────▼────────┐
       │   LSTM Decoder │   ← input = [embedding ⊕ context]
       │  (512 units)   │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ Dense + Softmax│   → token probabilities
       └────────────────┘
```

## Feature Engineering

| Feature | Description |
|---|---|
| `urdu_clean` | Diacritics removed, punctuation spaced |
| `english_clean` | Lowercased, non-ascii stripped |
| `urdu_len` / `english_len` | Token counts |
| `length_ratio` | EN/UR token ratio |
| `is_question` | Binary: ends with ؟ / ? |
| `is_exclamation` | Binary: ends with ! |
| `avg_word_len_*` | Mean chars per token |
| `unique_*_tokens` | Type-token ratio proxy |

## Dataset

110 parallel Urdu–English sentences across 10 domains:
- Greetings & social · Daily life · Family · Food · Emotions
- Nature · Time · Education · Places · Health

## Project structure

```
urdu_english_mt/
├── dataset.py          # dataset generator + feature engineering
├── model.py            # Keras encoder-decoder-attention model
├── app.py              # Streamlit frontend
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Option A – train then launch UI
python model.py          # trains + saves to model_artifacts/
streamlit run app.py

# Option B – launch UI and train from the sidebar button
streamlit run app.py
```

## Streamlit tabs

| Tab | Contents |
|---|---|
| 🌐 Translate | Enter Urdu → get English translation |
| 📊 Dataset | Browse the full dataset with domain filters & charts |
| 📈 Training Curves | Loss / accuracy per epoch |
| 🔍 Attention Map | Bahdanau attention heatmap for last translation |

"""
Streamlit Frontend: Urdu ↔ English Neural Machine Translation
Run: streamlit run app.py
"""

import os, sys, pickle, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="اردو ↔ English NMT",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS: Deep Indigo / Amber editorial aesthetic ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
}

/* Background */
.stApp {
    background: #0f0e17;
    color: #fffffe;
}

/* Main header */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f7b731 0%, #fd9644 50%, #fc5c65 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #a7a9be;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(247,183,49,0.2);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Translation output box */
.translation-box {
    background: linear-gradient(135deg, rgba(247,183,49,0.08), rgba(252,92,101,0.06));
    border-left: 4px solid #f7b731;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.5rem;
    font-size: 1.35rem;
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    color: #fffffe;
    letter-spacing: 0.01em;
    line-height: 1.6;
    margin: 1rem 0;
}

.urdu-text {
    font-size: 1.5rem;
    direction: rtl;
    text-align: right;
    line-height: 2;
    color: #f7b731;
    font-weight: 600;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(247,183,49,0.2);
    margin-bottom: 1rem;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: rgba(247,183,49,0.1);
    border: 1px solid rgba(247,183,49,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #f7b731;
    margin: 0.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0c1d;
    border-right: 1px solid rgba(247,183,49,0.15);
}

[data-testid="stSidebar"] * {
    color: #fffffe !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f7b731, #fd9644);
    color: #0f0e17 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s ease;
    text-transform: uppercase;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(247,183,49,0.35);
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(247,183,49,0.25) !important;
    border-radius: 8px !important;
    color: #fffffe !important;
    font-size: 1rem !important;
    font-family: 'Source Serif 4', serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(247,183,49,0.2);
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a7a9be !important;
    background: transparent;
    border: none;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    color: #f7b731 !important;
    border-bottom: 2px solid #f7b731 !important;
}

/* Divider */
hr { border-color: rgba(247,183,49,0.15) !important; }

/* Selectbox */
.stSelectbox select, [data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(247,183,49,0.25) !important;
    color: #fffffe !important;
}

.status-trained {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,200,100,0.1);
    border: 1px solid rgba(0,200,100,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #5eff9e;
}
.status-untrained {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,100,100,0.1);
    border: 1px solid rgba(255,100,100,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #ff6b6b;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #a7a9be;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Lazy imports (TF is slow) ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_tf():
    import tensorflow as tf
    from tensorflow import keras
    return tf, keras

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load trained model from disk; returns None if not trained yet."""
    save_dir = "model_artifacts"
    if not os.path.exists(os.path.join(save_dir, "config.pkl")):
        return None
    try:
        from model import load_model_artifacts
        enc, dec, sv, tv, sm, cfg = load_model_artifacts(save_dir)
        return enc, dec, sv, tv, sm, cfg
    except Exception as e:
        return str(e)

def run_training():
    """Train the model (called from UI)."""
    from model import train_model
    with st.spinner("Training model — this may take a few minutes on CPU…"):
        enc, dec, sv, tv, sm, tmax, hist = train_model()
    st.cache_resource.clear()
    return hist

# ─── Header ──────────────────────────────────────────────────────────────────
col_hdr, col_status = st.columns([3, 1])
with col_hdr:
    st.markdown('<div class="main-title">اردو ↔ English</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Neural Machine Translation · BiLSTM Encoder · Attention Decoder</div>',
                unsafe_allow_html=True)

artifacts = load_artifacts()
model_ready = artifacts is not None and not isinstance(artifacts, str)

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if model_ready:
        st.markdown('<span class="status-trained">● Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-untrained">● Not Trained</span>', unsafe_allow_html=True)

st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
<div class="glass-card">
<p style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#a7a9be;line-height:2">
Encoder → BiLSTM<br>
Attention → Bahdanau<br>
Decoder → LSTM × 1<br>
Embedding dim → 128<br>
LSTM units → 256<br>
Framework → Keras/TF<br>
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### ⚙️ Training")
    if st.button("🚀 Train Model"):
        try:
            hist = run_training()
            st.success(f"Training complete!  Final loss: {hist.history['loss'][-1]:.4f}")
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.markdown("---")
    st.markdown("### 📊 Dataset info")
    from dataset import build_dataset
    df_info = build_dataset()
    st.metric("Sentence pairs", len(df_info))
    st.metric("Avg Urdu tokens", f"{df_info['urdu_len'].mean():.1f}")
    st.metric("Avg English tokens", f"{df_info['english_len'].mean():.1f}")
    st.metric("Question sentences", int(df_info['is_question'].sum()))

# ─── Main tabs ───────────────────────────────────────────────────────────────
tabs = st.tabs(["🌐 Translate", "📊 Dataset", "📈 Training Curves", "🔍 Attention Map"])

# ── TAB 1: Translate ─────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-label">Input Urdu sentence</div>', unsafe_allow_html=True)

    # Quick examples
    EXAMPLES = [
        "میں خوش ہوں۔",
        "آج موسم اچھا ہے۔",
        "مجھے بریانی پسند ہے۔",
        "وہ کتاب پڑھ رہا ہے۔",
        "بارش ہو رہی ہے۔",
        "آپ کیسے ہیں؟",
        "درخت بڑے ہیں۔",
        "محنت کامیابی کی چابی ہے۔",
    ]
    ex_col1, ex_col2 = st.columns([2, 1])
    with ex_col2:
        selected_ex = st.selectbox("Quick examples", ["— pick one —"] + EXAMPLES, label_visibility="collapsed")

    with ex_col1:
        default_val = selected_ex if selected_ex != "— pick one —" else ""
        urdu_input = st.text_area(
            "Urdu sentence",
            value=default_val,
            height=100,
            placeholder="یہاں اردو جملہ لکھیں…",
            label_visibility="collapsed",
        )

    translate_btn = st.button("Translate →")

    if translate_btn:
        if not urdu_input.strip():
            st.warning("Please enter an Urdu sentence.")
        elif not model_ready:
            st.error("Model not trained yet. Click **Train Model** in the sidebar.")
        else:
            enc, dec, sv, tv, sm, cfg = artifacts
            from model import translate
            with st.spinner("Translating…"):
                t0 = time.time()
                translation, attn = translate(urdu_input, enc, dec, sv, tv, sm)
                elapsed = time.time() - t0

            st.markdown('<div class="section-label">Source (Urdu)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="urdu-text">{urdu_input}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label">Translation (English)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="translation-box">{translation if translation else "…"}</div>',
                        unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            src_toks = len(urdu_input.split())
            tgt_toks = len(translation.split()) if translation else 0
            m1.metric("Source tokens", src_toks)
            m2.metric("Output tokens", tgt_toks)
            m3.metric("Inference time", f"{elapsed*1000:.1f} ms")

            # Store attn for tab 4
            st.session_state["last_attn"] = attn
            st.session_state["last_src"]  = urdu_input.split()
            st.session_state["last_tgt"]  = translation.split() if translation else []

# ── TAB 2: Dataset ────────────────────────────────────────────────────────────
with tabs[1]:
    from dataset import build_dataset
    df = build_dataset()

    st.markdown("### Feature-Engineered Dataset")
    st.markdown(f"<span class='metric-chip'>Total pairs: {len(df)}</span>"
                f"<span class='metric-chip'>Domains: 10</span>"
                f"<span class='metric-chip'>Languages: Urdu + English</span>",
                unsafe_allow_html=True)

    # Domain filter
    domain_map = {
        "All": slice(None),
        "Greetings": slice(0, 21),
        "Daily Life": slice(21, 41),
        "Family": slice(41, 51),
        "Food": slice(51, 61),
        "Emotions": slice(61, 71),
        "Nature": slice(71, 81),
        "Time": slice(81, 91),
        "Education": slice(91, 99),
        "Places": slice(99, 105),
        "Health": slice(105, 110),
    }
    domain = st.selectbox("Filter by domain", list(domain_map.keys()))
    subset = df.iloc[domain_map[domain]] if domain != "All" else df

    display_cols = ["urdu", "english", "urdu_len", "english_len",
                    "length_ratio", "is_question", "avg_word_len_urdu"]
    st.dataframe(
        subset[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # Distribution chart
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Urdu vs English token length distribution**")
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0f0e17")
        ax.set_facecolor("#0f0e17")
        ax.hist(df["urdu_len"], bins=12, color="#f7b731", alpha=0.75, label="Urdu")
        ax.hist(df["english_len"], bins=12, color="#fc5c65", alpha=0.75, label="English")
        ax.set_xlabel("Token count", color="#a7a9be")
        ax.set_ylabel("Frequency", color="#a7a9be")
        ax.tick_params(colors="#a7a9be")
        ax.legend(facecolor="#1a1a2e", labelcolor="#fffffe")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        st.pyplot(fig)
    with c2:
        st.markdown("**Length ratio (English/Urdu tokens)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor="#0f0e17")
        ax2.set_facecolor("#0f0e17")
        ax2.scatter(df["urdu_len"], df["english_len"], color="#f7b731", alpha=0.6, s=30)
        ax2.set_xlabel("Urdu tokens", color="#a7a9be")
        ax2.set_ylabel("English tokens", color="#a7a9be")
        ax2.tick_params(colors="#a7a9be")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")
        st.pyplot(fig2)

# ── TAB 3: Training Curves ────────────────────────────────────────────────────
with tabs[2]:
    save_dir = "model_artifacts"
    if os.path.exists(os.path.join(save_dir, "config.pkl")):
        with open(os.path.join(save_dir, "config.pkl"), "rb") as f:
            cfg = pickle.load(f)
        loss_hist = cfg.get("history_loss", [])
        acc_hist  = cfg.get("history_acc", [])

        if loss_hist:
            fig3, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0f0e17")
            for ax in axes:
                ax.set_facecolor("#0f0e17")
                ax.tick_params(colors="#a7a9be")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333")

            epochs_range = range(1, len(loss_hist) + 1)
            axes[0].plot(epochs_range, loss_hist, color="#f7b731", linewidth=2)
            axes[0].fill_between(epochs_range, loss_hist, alpha=0.1, color="#f7b731")
            axes[0].set_title("Training Loss", color="#fffffe", fontsize=12)
            axes[0].set_xlabel("Epoch", color="#a7a9be")
            axes[0].set_ylabel("Masked Cross-Entropy Loss", color="#a7a9be")

            if acc_hist:
                axes[1].plot(epochs_range, acc_hist, color="#fc5c65", linewidth=2)
                axes[1].fill_between(epochs_range, acc_hist, alpha=0.1, color="#fc5c65")
                axes[1].set_title("Training Accuracy", color="#fffffe", fontsize=12)
                axes[1].set_xlabel("Epoch", color="#a7a9be")
                axes[1].set_ylabel("Accuracy", color="#a7a9be")
            else:
                axes[1].text(0.5, 0.5, "Accuracy not recorded",
                             ha="center", va="center", color="#a7a9be", transform=axes[1].transAxes)

            plt.tight_layout()
            st.pyplot(fig3)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Final Loss",  f"{loss_hist[-1]:.4f}")
            col_b.metric("Best Loss",   f"{min(loss_hist):.4f}")
            col_c.metric("Epochs Run",  len(loss_hist))
        else:
            st.info("No training history found.")
    else:
        st.info("Train the model first to see curves.")

# ── TAB 4: Attention Map ──────────────────────────────────────────────────────
with tabs[3]:
    if "last_attn" in st.session_state and len(st.session_state["last_tgt"]) > 0:
        attn  = st.session_state["last_attn"]
        src_w = st.session_state["last_src"]
        tgt_w = st.session_state["last_tgt"]

        # Trim attention to actual token counts
        attn_trim = attn[:len(tgt_w), :len(src_w)]

        fig4, ax4 = plt.subplots(
            figsize=(max(6, len(src_w) * 0.8 + 2), max(4, len(tgt_w) * 0.6 + 2)),
            facecolor="#0f0e17"
        )
        ax4.set_facecolor("#1a1a2e")
        im = ax4.imshow(attn_trim, cmap="YlOrRd", aspect="auto", vmin=0, vmax=attn_trim.max())
        ax4.set_xticks(range(len(src_w)))
        ax4.set_yticks(range(len(tgt_w)))
        ax4.set_xticklabels(src_w, rotation=45, ha="right", color="#f7b731", fontsize=9)
        ax4.set_yticklabels(tgt_w, color="#fffffe", fontsize=9)
        ax4.set_xlabel("Source (Urdu)", color="#a7a9be")
        ax4.set_ylabel("Target (English)", color="#a7a9be")
        ax4.set_title("Bahdanau Attention Weights", color="#fffffe", fontsize=12, pad=15)
        plt.colorbar(im, ax=ax4, fraction=0.03).ax.tick_params(colors="#a7a9be")
        plt.tight_layout()
        st.pyplot(fig4)

        st.markdown("""
<div class="glass-card">
<p style="font-size:0.88rem;color:#a7a9be;line-height:1.8">
<b style="color:#f7b731">How to read this:</b> Each row is a decoded English word. Each column is an Urdu source token.
Brighter cells mean the decoder paid more <em>attention</em> to that source token while generating that English word.
Diagonal alignment patterns indicate the model has learnt word-level correspondences.
</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Run a translation first (Tab 1) to see the attention heatmap here.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#3d3d5c;text-align:center">'
    'BiLSTM Encoder · Bahdanau Attention · LSTM Decoder · Keras 3 / TensorFlow 2 · Streamlit'
    '</p>',
    unsafe_allow_html=True,
)

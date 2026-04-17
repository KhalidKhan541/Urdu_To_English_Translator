import os, pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from dataset import build_dataset, clean_text

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
EMBEDDING_DIM = 128
LSTM_UNITS    = 256
BATCH_SIZE    = 16
EPOCHS        = 50

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# ─────────────────────────────────────
# VOCAB
# ─────────────────────────────────────
class Vocabulary:
    def __init__(self, name, max_vocab):
        self.name      = name
        self.max_vocab = max_vocab
        self.word2idx  = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2word  = {v: k for k, v in self.word2idx.items()}
        self.freq      = {}

    @property
    def size(self):
        return len(self.word2idx)

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.freq[w] = self.freq.get(w, 0) + 1

    def build(self):
        sorted_words = sorted(self.freq, key=self.freq.get, reverse=True)
        for w in sorted_words[: self.max_vocab - len(self.word2idx)]:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, sentence):
        return [
            self.word2idx.get(w, self.word2idx[UNK_TOKEN])
            for w in sentence.split()
        ]

    def decode(self, ids):
        words = [self.idx2word.get(i, UNK_TOKEN) for i in ids]
        return " ".join(
            w for w in words if w not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
        )


# ─────────────────────────────────────
# DATA
# ─────────────────────────────────────
def prepare_data(df):
    src_vocab = Vocabulary("urdu",    3000)
    tgt_vocab = Vocabulary("english", 3000)

    for _, r in df.iterrows():
        src_vocab.add_sentence(r["urdu_clean"])
        tgt_vocab.add_sentence(r["english_clean"])

    src_vocab.build()
    tgt_vocab.build()

    src_seqs = [src_vocab.encode(r["urdu_clean"]) for _, r in df.iterrows()]
    tgt_seqs = [
        [tgt_vocab.word2idx[SOS_TOKEN]]
        + tgt_vocab.encode(r["english_clean"])
        + [tgt_vocab.word2idx[EOS_TOKEN]]
        for _, r in df.iterrows()
    ]

    src_max = max(len(x) for x in src_seqs)
    tgt_max = max(len(x) for x in tgt_seqs)

    src_pad = keras.preprocessing.sequence.pad_sequences(
        src_seqs, maxlen=src_max, padding="post"
    )
    tgt_pad = keras.preprocessing.sequence.pad_sequences(
        tgt_seqs, maxlen=tgt_max, padding="post"
    )

    dec_input  = tgt_pad[:, :-1]
    dec_target = tgt_pad[:, 1:]

    return src_pad, dec_input, dec_target, src_vocab, tgt_vocab, src_max, tgt_max


# ─────────────────────────────────────
# ATTENTION
# ─────────────────────────────────────
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.V  = layers.Dense(1,     use_bias=False)

    def call(self, query, values):
        query   = keras.ops.expand_dims(query, axis=1)
        score   = self.V(keras.ops.tanh(self.W1(values) + self.W2(query)))
        weights = keras.ops.softmax(score, axis=1)
        context = keras.ops.sum(weights * values, axis=1)
        return context, keras.ops.squeeze(weights, axis=-1)


# ─────────────────────────────────────
# ENCODER
# ─────────────────────────────────────
def build_encoder(vocab_size, emb, units, maxlen):
    inp = keras.Input(shape=(maxlen,))
    x   = layers.Embedding(vocab_size, emb, mask_zero=True)(inp)

    lstm = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True, return_state=True)
    )
    enc_out, fwd_h, fwd_c, bwd_h, bwd_c = lstm(x)

    # Merge fwd+bwd states -> size units*2 to initialise decoder
    h = layers.Concatenate()([fwd_h, bwd_h])
    c = layers.Concatenate()([fwd_c, bwd_c])

    return Model(inp, [enc_out, h, c], name="encoder")


# ─────────────────────────────────────
# DECODER CELL
# ─────────────────────────────────────
class DecoderCell(layers.Layer):
    def __init__(self, vocab_size, emb, units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, emb)
        self.attention = BahdanauAttention(units)
        # FIX: units*2 to match encoder concatenated state size
        self.lstm   = layers.LSTM(units * 2, return_state=True, return_sequences=False)
        self.concat = layers.Concatenate()
        self.fc     = layers.Dense(vocab_size, activation="softmax")

    def call(self, token, enc_out, h, c):
        emb        = self.embedding(token)
        context, _ = self.attention(h, enc_out)
        context    = keras.ops.expand_dims(context, 1)
        x          = self.concat([emb, context])
        out, h, c  = self.lstm(x, initial_state=[h, c])
        return self.fc(out), h, c


# ─────────────────────────────────────
# BUILD TRAINING MODEL
# ─────────────────────────────────────
def build_model(src_vocab_size, tgt_vocab_size, src_max, tgt_max):
    enc_in = keras.Input(shape=(src_max,))
    dec_in = keras.Input(shape=(tgt_max - 1,))

    encoder = build_encoder(src_vocab_size, EMBEDDING_DIM, LSTM_UNITS, src_max)
    enc_out, h, c = encoder(enc_in)

    decoder = DecoderCell(tgt_vocab_size, EMBEDDING_DIM, LSTM_UNITS)

    outputs = []
    for t in range(tgt_max - 1):
        token = dec_in[:, t:t+1]
        out, h, c = decoder(token, enc_out, h, c)
        outputs.append(keras.ops.expand_dims(out, 1))

    outputs = keras.layers.Concatenate(axis=1)(outputs)
    model   = Model([enc_in, dec_in], outputs)
    return model, encoder, decoder


# ─────────────────────────────────────
# LOSS
# ─────────────────────────────────────
def masked_loss(y_true, y_pred):
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = keras.ops.cast(keras.ops.not_equal(y_true, 0), "float32")
    return keras.ops.sum(loss * mask) / keras.ops.sum(mask)


# ─────────────────────────────────────
# TRANSLATE (local sanity check)
# ─────────────────────────────────────
def translate(text, encoder, decoder, src_vocab, tgt_vocab, src_maxlen, max_len=30):
    cleaned = clean_text(text, "ur")
    ids     = src_vocab.encode(cleaned)
    padded  = keras.preprocessing.sequence.pad_sequences(
        [ids], maxlen=src_maxlen, padding="post"
    )
    enc_out, h, c = encoder(tf.convert_to_tensor(padded), training=False)
    token  = tf.constant([[tgt_vocab.word2idx[SOS_TOKEN]]])
    result = []
    for _ in range(max_len):
        logits, h, c = decoder(token, enc_out, h, c)
        pred = int(tf.argmax(logits, axis=-1).numpy()[0])
        if pred == tgt_vocab.word2idx[EOS_TOKEN]:
            break
        result.append(pred)
        token = tf.constant([[pred]])
    return tgt_vocab.decode(result)


# ─────────────────────────────────────
# TRAIN
# ─────────────────────────────────────
def train_model():
    df = build_dataset()

    src, dec_in, dec_out, src_vocab, tgt_vocab, src_max, tgt_max = prepare_data(df)

    model, encoder, decoder = build_model(
        src_vocab.size, tgt_vocab.size, src_max, tgt_max
    )

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=masked_loss)
    model.summary()

    # FIX: capture history for Streamlit training curves tab
    history = model.fit(
        [src, dec_in], dec_out,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    os.makedirs("model_artifacts", exist_ok=True)

    # Encoder -> .h5
    encoder.save_weights("model_artifacts/encoder.weights.h5")

    # FIX: decoder -> .pkl (matches model_inference.py loader)
    with open("model_artifacts/decoder_weights.pkl", "wb") as f:
        pickle.dump(decoder.get_weights(), f)

    # FIX: consistent key names + all keys inference needs
    with open("model_artifacts/config.pkl", "wb") as f:
        pickle.dump({
            "src_vocab":     src_vocab,
            "tgt_vocab":     tgt_vocab,
            "src_maxlen":    src_max,
            "tgt_maxlen":    tgt_max,
            "embedding_dim": EMBEDDING_DIM,
            "lstm_units":    LSTM_UNITS,
            "history_loss":  history.history["loss"],
            "history_acc":   history.history.get("accuracy", []),
        }, f)

    print("\n✅ Training complete. Artifacts saved to model_artifacts/")

    sample = df["urdu"].iloc[0]
    pred   = translate(sample, encoder, decoder, src_vocab, tgt_vocab, src_max)
    print(f"Sample   : {sample}")
    print(f"Predicted: {pred}")
    print(f"Expected : {df['english'].iloc[0]}")


if __name__ == "__main__":
    train()
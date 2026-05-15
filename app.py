import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- LOAD MODEL ----------
model = load_model("emotion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

MAX_LEN = 50

# ---------- TEXT CLEAN ----------
def normalize(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

# ---------- EMOTION ----------
def predict_emotion(sentence):
    cleaned = normalize(sentence)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(pad, verbose=0)[0]
    mood_index = np.argmax(pred)
    emotion_map = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }
    mood = emotion_map[mood_index]
    confidence = float(np.max(pred))
    probabilities = {}
    for i, p in enumerate(pred):
        emotion = emotion_map[i]
        probabilities[emotion] = round(float(p) * 100, 2)
    return mood, confidence, probabilities

# ---------- MOVIE DATA ----------
df = pd.read_csv("movie_dataset.csv")
features = ['keywords', 'cast', 'genres', 'director']
for f in features:
    df[f] = df[f].fillna('')
df["combined"] = (
    df['keywords'] + " " +
    df['cast'] + " " +
    df['genres'] + " " +
    df['director']
)
cv = CountVectorizer()
matrix = cv.fit_transform(df["combined"])
cosine_sim = cosine_similarity(matrix)

# ---------- MOOD → GENRE ----------
mood_to_genre = {
    'sadness': 'Drama',
    'joy': 'Comedy',
    'love': 'Romance',
    'anger': 'Action',
    'fear': 'Horror',
    'surprise': 'Adventure'
}

# ---------- RECOMMENDER ----------
def clean_title(t):
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9]', '', t)
    return t

def recommend(title, genre):
    title_clean = clean_title(title)
    df['title_clean'] = df['title'].apply(clean_title)
    if title_clean not in df['title_clean'].values:
        return ["Movie not found 😢"]
    idx = df[df['title_clean'] == title_clean].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rec = []
    for i in scores:
        movie_genre = df.iloc[i[0]]['genres']
        movie_title = df.iloc[i[0]]['title']
        if genre in movie_genre and clean_title(movie_title) != title_clean:
            rec.append(movie_title)
        if len(rec) == 5:
            break
    return rec

# ============================================================
# UI — STYLING ONLY (zero logic changes above)
# ============================================================

st.set_page_config(
    page_title="Moodplex · Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ---------- GLOBAL CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── kill streamlit top padding ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
}
/* hide the sticky header bar that eats ~60px */
[data-testid="stHeader"] {
    display: none !important;
}

/* ── root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #0a0a0f;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── global typography ── */
body, p, div, span, label {
    font-family: 'DM Sans', sans-serif !important;
    color: #e8e4dc !important;
}

/* ── hero block ── */
.hero {
    text-align: center;
    padding: 0.4rem 0 0.4rem;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #c8a96e !important;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(1.8rem, 4vw, 2.6rem) !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    color: #f5f0e8 !important;
    margin: 0 !important;
    display: block;
}
.hero-title span {
    background: linear-gradient(135deg, #c8a96e 0%, #e8c98a 50%, #c8a96e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Playfair Display', serif !important;
    font-weight: 900 !important;
}

/* ── divider ── */
.divider {
    border: none;
    border-top: 1px solid #1e1c1a;
    margin: 0.5rem 0;
}

/* ── section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a4845 !important;
    margin-bottom: 0.5rem;
}

/* ── inputs ── */
[data-testid="stTextInput"] input {
    background: #111118 !important;
    border: 1px solid #252420 !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label {
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5a5652 !important;
    font-weight: 500 !important;
    margin-bottom: 0.3rem !important;
}

/* ── quick pick buttons ── */
[data-testid="stButton"] > button {
    background: #111118 !important;
    border: 1px solid #1e1c1a !important;
    border-radius: 8px !important;
    color: #9a9490 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    padding: 0.5rem 0.4rem !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stButton"] > button:hover {
    background: #1a1814 !important;
    border-color: #c8a96e !important;
    color: #c8a96e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(200,169,110,0.15) !important;
}

/* ── CTA / primary button ── */
button[kind="primary"],
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #c8a96e, #e2b96a) !important;
    color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    height: auto !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: background 0.18s !important;
    box-shadow: 0 4px 20px rgba(200,169,110,0.3) !important;
    transform: none !important;
}
button[kind="primary"]:hover,
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #e8c98a, #f5d98a) !important;
    transform: none !important;
    opacity: 1 !important;
    color: #0a0a0f !important;
}

/* ── emotion pill ── */
.emotion-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(200,169,110,0.15), rgba(200,169,110,0.05));
    border: 1px solid rgba(200,169,110,0.35);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #c8a96e !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.4rem 0;
}

/* ── progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #c8a96e, #e8c98a) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div {
    background: #1a1814 !important;
    border-radius: 999px !important;
    height: 6px !important;
}

/* ── genre badge ── */
.genre-badge {
    display: inline-block;
    background: #111118;
    border: 1px solid #252420;
    border-radius: 6px;
    padding: 0.35rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: #7a7672 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── movie card list ── */
.movie-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #0f0f16;
    border: 1px solid #1a1814;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.4rem;
    transition: border-color 0.18s, transform 0.18s;
}
.movie-card:hover {
    border-color: #2a2820;
    transform: translateX(3px);
}
.movie-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 900;
    color: #c8a96e !important;
    min-width: 2rem;
    text-align: center;
    opacity: 0.55;
}
.movie-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    color: #d8d4cc !important;
}

/* ── warning / info ── */
[data-testid="stAlert"] {
    background: #111118 !important;
    border: 1px solid #252420 !important;
    border-radius: 10px !important;
    color: #7a7672 !important;
}

/* ── force black text inside primary buttons always ── */
[data-testid="stButton"] > button[kind="primary"] p,
[data-testid="stButton"] > button[kind="primary"]:hover p {
    color: #0a0a0f !important;
}
h1 a, h2 a, h3 a, .hero-title a { display: none !important; }
[data-testid="stMarkdownContainer"] h1 { pointer-events: none; }

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.element-container { margin-bottom: 0 !important; }
[data-testid="stMarkdownContainer"] p { margin: 0 !important; }



.result-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f5f0e8 !important;
    margin: 0.5rem 0 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <p class="hero-eyebrow">AI · Cinema · Emotion</p>
    <div class="hero-title">Watch what you <span>feel</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- SESSION ----------
if "movie_name" not in st.session_state:
    st.session_state.movie_name = ""
if "results" not in st.session_state:
    st.session_state.results = None

def set_movie(name):
    st.session_state.movie_name = name

# ---------- RESULTS ZONE
if st.session_state.results:
    r = st.session_state.results
    mood_emoji = {
        'sadness': '💙', 'joy': '✨', 'love': '❤️',
        'anger': '🔥', 'fear': '👁', 'surprise': '⚡'
    }
    emoji = mood_emoji.get(r["mood"], "🎭")
    st.markdown(
        f'<div style="text-align:center;margin:0.1rem 0 0.3rem">'
        f'<div class="emotion-pill">{emoji} {r["mood"].upper()} detected</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="text-align:center;margin:0 0 0.6rem">'
        f'<span class="genre-badge">Serving · {r["genre"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('<p class="result-header">Top Picks For You</p>', unsafe_allow_html=True)
    if r["movies"] and r["movies"][0] != "Movie not found 😢":
        for idx, m in enumerate(r["movies"], 1):
            st.markdown(
                f'<div class="movie-card">'
                f'  <span class="movie-rank">0{idx}</span>'
                f'  <span class="movie-title">{m}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Couldn't find recommendations for that title. Try a different movie.")

    # Slim reset — no full form repeat
    if st.button("↩ Try a different mood", type="primary"):
        st.session_state.results = None
        st.session_state.movie_name = ""
        st.rerun()

else:
    # ---------- INPUT FIELDS ----------
    sentence = st.text_input(
        "How are you feeling right now?",
        placeholder="e.g. I feel really lonely and miss my old friends…"
    )

    st.text_input(
        "A movie you already enjoy",
        placeholder="e.g. Inception, Titanic, The Dark Knight…",
        key="movie_name"
    )

    # ---------- QUICK PICKS ----------
    st.markdown('<p class="section-label">Quick picks</p>', unsafe_allow_html=True)

    quick_movies = [
        "Avatar", "Titanic", "Inception", "The Dark Knight",
        "Interstellar", "Gladiator", "The Hangover", "Frozen"
    ]

    cols = st.columns(4)
    for i, m in enumerate(quick_movies):
        cols[i % 4].button(m, on_click=set_movie, args=(m,))

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # ---------- CTA ----------
    if st.button("✦ Discover My Films", type="primary"):
        movie = st.session_state.movie_name

        if sentence.strip() == "" or movie.strip() == "":
            st.warning("Please fill in both fields to continue.")
        else:
            # ---- LOGIC (untouched) ----
            mood, conf, probs = predict_emotion(sentence)
            genre = mood_to_genre[mood]
            movies = recommend(movie, genre)

            st.session_state.results = {
                "mood": mood,
                "genre": genre,
                "movies": movies
            }
            st.rerun()
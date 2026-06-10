import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from urllib.parse import urlparse, parse_qs
from collections import Counter

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen – Manga Evo",
    page_icon="📚",
    layout="wide",
)

# ── NLTK Downloads ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

download_nltk()

# ── Preprocessing ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_stopwords_and_lemmatizer():
    indonesian_stopwords = {
        "dan","nya","yang","di","ke","dari","untuk","pada","ini","itu",
        "atau","sebagai","dengan","karena","adalah","adanya","saja","lagu",
        "lebih","sangat","tidak","bukan","oleh","agar","kami","kamu","mereka",
        "saya","kita","anda","dia","apa","siapa","bagaimana","mengapa",
        "dimana","bisa","ada",
    }
    stop_words = set(stopwords.words("english")).union(indonesian_stopwords)
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = get_stopwords_and_lemmatizer()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)


def get_package_name(play_store_url):
    parsed = urlparse(play_store_url)
    params = parse_qs(parsed.query)
    return params.get("id", [None])[0]


# ── Scraping (cached) ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def scrape_reviews(package_name, count=1000):
    from google_play_scraper import reviews, Sort
    app_reviews, _ = reviews(
        package_name,
        lang="id",
        country="id",
        count=count,
        sort=Sort.NEWEST,
    )
    df = pd.DataFrame(app_reviews)
    return df[["reviewId", "content", "score"]].copy()


# ── Model Training (cached) ────────────────────────────────────────────────────
@st.cache_resource
def train_model(df_hash):
    # This function is keyed by a hash, so it retrains only when data changes
    return None  # placeholder – real call below


@st.cache_data(show_spinner=False)
def build_model(df_json):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    df = pd.read_json(df_json, orient="records")
    df["cleaned_text"] = df["content"].apply(preprocess_text)
    df["sentiment"] = df["score"].apply(lambda x: 1 if x > 3 else 0)

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, use_idf=True, norm="l2", smooth_idf=True)
    X = tfidf.fit_transform(df["cleaned_text"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negatif", "Positif"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    all_preds = model.predict(X)
    df["predicted_sentiment"] = all_preds
    df["sentiment_label"] = df["predicted_sentiment"].map({1: "Positif", 0: "Negatif"})

    return model, tfidf, df, acc, report, cm


def predict_sentiment(text, model, tfidf):
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "Positif ✅" if pred == 1 else "Negatif ❌"
    conf = max(proba) * 100
    return label, conf


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📚 Analisis Sentimen Ulasan Aplikasi Manga Evo")
st.caption("Mata Kuliah Eksplorasi dan Visualisasi Data · Statistika Universitas Matana")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    url_input = st.text_input(
        "Google Play Store URL",
        value="https://play.google.com/store/apps/details?id=com.mangaid.bacakomikid&hl=id",
    )
    count_input = st.slider("Jumlah ulasan yang diambil", 100, 1000, 1000, step=100)
    run_btn = st.button("🚀 Ambil Data & Analisis", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Info Aplikasi**")
    st.markdown("- Rating rata-rata: **2.9 / 5**")
    st.markdown("- Bahasa: Indonesia 🇮🇩")
    st.markdown("- Model: Logistic Regression + TF-IDF")

# ── Session State ──────────────────────────────────────────────────────────────
if "df_result" not in st.session_state:
    st.session_state.df_result = None
    st.session_state.model = None
    st.session_state.tfidf = None
    st.session_state.acc = None
    st.session_state.report = None
    st.session_state.cm = None

# ── Main Pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    pkg = get_package_name(url_input)
    if not pkg:
        st.error("URL tidak valid. Pastikan URL mengandung parameter `id=`.")
    else:
        with st.spinner(f"Mengambil {count_input} ulasan dari Google Play Store…"):
            try:
                df_raw = scrape_reviews(pkg, count_input)
                st.success(f"✅ Berhasil mengambil **{len(df_raw)}** ulasan.")
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")
                st.stop()

        with st.spinner("Memproses teks & melatih model…"):
            model, tfidf, df_result, acc, report, cm = build_model(df_raw.to_json(orient="records"))

        st.session_state.df_result = df_result
        st.session_state.model = model
        st.session_state.tfidf = tfidf
        st.session_state.acc = acc
        st.session_state.report = report
        st.session_state.cm = cm

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.df_result is not None:
    df = st.session_state.df_result
    model = st.session_state.model
    tfidf = st.session_state.tfidf
    acc = st.session_state.acc
    report = st.session_state.report
    cm = st.session_state.cm

    tabs = st.tabs(["📊 Ringkasan", "☁️ Word Cloud", "🤖 Model", "🔍 Prediksi Baru", "📋 Data Mentah"])

    # ── Tab 1: Ringkasan ─────────────────────────────────────────────────────
    with tabs[0]:
        import matplotlib.pyplot as plt
        import numpy as np

        st.subheader("Ringkasan Distribusi Sentimen")

        counts = df["sentiment_label"].value_counts()
        pos = counts.get("Positif", 0)
        neg = counts.get("Negatif", 0)
        total = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ulasan", total)
        col2.metric("Positif ✅", f"{pos} ({pos/total*100:.1f}%)")
        col3.metric("Negatif ❌", f"{neg} ({neg/total*100:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors_bar = ["#2ca02c" if x == "Positif" else "#d62728" for x in counts.index]
        bars = axes[0].bar(counts.index, counts.values, color=colors_bar, edgecolor="white")
        axes[0].set_title("Distribusi Prediksi Sentimen", fontsize=13)
        axes[0].set_xlabel("Sentimen"); axes[0].set_ylabel("Jumlah Ulasan")
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 5, int(h), ha="center", fontweight="bold")
        axes[0].set_ylim(0, max(counts.values) * 1.15)

        axes[1].pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#2ca02c", "#d62728"], startangle=90, explode=[0.05] * len(counts),
            textprops={"fontsize": 12},
        )
        axes[1].set_title("Proporsi Sentimen", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Distribusi Rating Bintang")
        star_counts = df["score"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.bar(star_counts.index.astype(str), star_counts.values, color="#5B8DB8")
        ax2.set_xlabel("Rating (Bintang)"); ax2.set_ylabel("Jumlah")
        ax2.set_title("Distribusi Rating Pengguna")
        for i, v in enumerate(star_counts.values):
            ax2.text(i, v + 2, str(v), ha="center", fontweight="bold")
        st.pyplot(fig2)

    # ── Tab 2: Word Cloud ────────────────────────────────────────────────────
    with tabs[1]:
        from wordcloud import WordCloud
        from matplotlib.cm import viridis_r
        from matplotlib.colors import Normalize

        st.subheader("Word Cloud – Kata Paling Sering Muncul")
        all_text = " ".join(df["cleaned_text"].dropna())

        wc = WordCloud(width=900, height=400, background_color="white", colormap="viridis", max_words=100)
        wc.generate(all_text)
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.imshow(wc, interpolation="bilinear")
        ax3.axis("off")
        ax3.set_title("Word Cloud – Ulasan Aplikasi Manga Evo", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig3)

        st.subheader("Top 20 Kata Paling Sering Muncul")
        all_words = all_text.split()
        word_counts = Counter(all_words)
        wf_df = (
            pd.DataFrame(word_counts.items(), columns=["Kata", "Frekuensi"])
            .sort_values("Frekuensi", ascending=False)
            .reset_index(drop=True)
        )
        top20 = wf_df.head(20)
        norm = Normalize(vmin=top20["Frekuensi"].min(), vmax=top20["Frekuensi"].max())
        colors = [viridis_r(norm(v)) for v in top20["Frekuensi"]]

        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.bar(top20["Kata"], top20["Frekuensi"], color=colors)
        ax4.set_xlabel("Kata"); ax4.set_ylabel("Frekuensi")
        ax4.set_title("Top 20 Kata dalam Ulasan Manga Evo")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig4)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(wf_df.head(20), use_container_width=True)

    # ── Tab 3: Model ─────────────────────────────────────────────────────────
    with tabs[2]:
        import numpy as np

        st.subheader("Evaluasi Model Logistic Regression")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{acc*100:.2f}%")
        col2.metric("Precision (Positif)", f"{report['Positif']['precision']*100:.1f}%")
        col3.metric("Recall (Positif)", f"{report['Positif']['recall']*100:.1f}%")
        col4.metric("F1-Score (Positif)", f"{report['Positif']['f1-score']*100:.1f}%")

        st.subheader("Confusion Matrix")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        im = ax5.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax5)
        ax5.set_xticks([0, 1]); ax5.set_xticklabels(["Negatif", "Positif"])
        ax5.set_yticks([0, 1]); ax5.set_yticklabels(["Negatif", "Positif"])
        for i in range(2):
            for j in range(2):
                ax5.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax5.set_xlabel("Prediksi"); ax5.set_ylabel("Aktual")
        ax5.set_title("Confusion Matrix – Logistic Regression")
        plt.tight_layout()
        st.pyplot(fig5)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # ── Tab 4: Prediksi Baru ─────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🔍 Coba Prediksi Sentimen Teks Baru")
        user_input = st.text_area(
            "Masukkan teks ulasan:",
            placeholder="Contoh: Aplikasi ini sering crash dan loading lama banget…",
            height=120,
        )
        if st.button("Prediksi Sentimen", type="primary"):
            if user_input.strip():
                label, conf = predict_sentiment(user_input, model, tfidf)
                st.markdown(f"### Hasil: {label}")
                st.progress(int(conf))
                st.caption(f"Confidence: **{conf:.1f}%**")
                with st.expander("Lihat teks setelah preprocessing"):
                    st.code(preprocess_text(user_input))
            else:
                st.warning("Masukkan teks terlebih dahulu.")

        st.divider()
        st.subheader("Contoh Kalimat Uji")
        examples = [
            "semangat ya untuk pengembangannya, semoga kedepannya bisa lebih baik lagi",
            "loading screen yang sangat lama, aplikasinya sering error dan crash",
            "komiknya lengkap, tampilannya bagus dan mudah digunakan",
            "tidak bisa dibuka sama sekali, sudah uninstall berkali-kali",
        ]
        for sent in examples:
            lbl, conf = predict_sentiment(sent, model, tfidf)
            color = "🟢" if "Positif" in lbl else "🔴"
            st.markdown(f"{color} **{lbl}** ({conf:.1f}%) — *{sent}*")

    # ── Tab 5: Data Mentah ───────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Data Ulasan (1000 Teratas)")
        display_cols = ["content", "score", "cleaned_text", "sentiment_label"]
        st.dataframe(df[display_cols].head(100), use_container_width=True)
        csv = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh CSV", csv, "ulasan_manga_evo.csv", "text/csv")

else:
    st.info("👈 Klik **Ambil Data & Analisis** di sidebar untuk memulai.")
    st.markdown("""
    ### Tentang Aplikasi ini
    Aplikasi ini melakukan analisis sentimen terhadap ulasan pengguna aplikasi **Manga Evo**
    yang diambil langsung dari Google Play Store.

    **Pipeline:**
    1. 📥 Scraping ulasan via `google-play-scraper`
    2. 🧹 Text preprocessing (case folding, hapus noise, tokenisasi, stopwords, lemmatisasi)
    3. ☁️ Visualisasi Word Cloud & frekuensi kata
    4. 🤖 Pelatihan model Logistic Regression dengan TF-IDF
    5. 📊 Evaluasi & visualisasi hasil
    6. 🔍 Prediksi sentimen teks baru secara real-time

    ---
    *Tugas UAS · Eksplorasi dan Visualisasi Data · Statistika Universitas Matana 2025/2026*
    """)

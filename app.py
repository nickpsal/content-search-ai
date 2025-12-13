import os
import time
import streamlit as st
import base64
from pathlib import Path
from core import ImageSearcher, PDFSearcher, Model, AudioSearcher, CoreTools
import psutil

# ======================================================
# 🧠 STREAMLIT CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Search Content in Multimedia Digital Archives using AI",
    layout="wide"
)

# ======================================================
# 🎨 CUSTOM CSS STYLING
# ======================================================
st.markdown("""
<style>
/* DASHBOARD GRID & CARDS */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.dash-card {
    background: #141414;
    border-radius: 18px;
    border: 1px solid #2a2a2a;
    padding: 18px 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
    min-height: 120px;
}

.dash-card h3 {
    margin: 0 0 8px 0;
    font-size: 1.1rem;
}

.dash-card p {
    margin: 0;
    font-size: 0.9rem;
    color: #999;
}

[data-testid="stExpander"] {
    background-color: #141414;
    padding: 0;
    border-radius: 16px;
    border: 1px solid #2a2a2a;
    margin-bottom: 5px !important;
    box-shadow: 0 0 25px rgba(0,0,0,0.5), inset 0 0 12px rgba(255,255,255,0.03);
}

[data-testid="stExpander"] > details {
    border-radius: 16px !important;
}

[data-testid="stExpanderDetails"] {
    padding: 20px;
}

.section-title h2 {
    margin-bottom: 0;
}
.section-title p {
    margin-top: -5px;
    color: #aaa;
}

/* CARD */
.search-card {
    max-width: 900px;
    margin: 0 auto;
    padding: 25px;
    background: #141414;
    border-radius: 18px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 0 35px rgba(0,0,0,0.45);
}

/* GRID */
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 25px;
}

/* IMAGE CARD */
.result-card {
    position: relative;
    background-color: #1b1b1b;
    border-radius: 16px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.result-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 25px rgba(255,255,255,0.18);
}

.result-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* BADGE */
.badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.8);
    padding: 5px 10px;
    font-size: 0.85rem;
    border-radius: 8px;
    color: #ffd700;
    font-weight: bold;
}

/* OVERLAY */
.overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 8px;
    background: linear-gradient(180deg, transparent, rgba(0,0,0,0.9));
    text-align: center;
    color: #ddd;
    font-size: 0.9rem;
}

/* ANIMATION */
.fade-in {
    animation: fadeIn 0.4s ease forwards;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 🟣 STYLE ONLY THE REAL INPUT FIELD */
div[data-testid="stTextInput"] input {
    background: #1c1c1c !important;
    border: 1px solid #2d2d2d !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    color: #e6e6e6 !important;
    font-size: 1.05rem !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.35) !important;
}

/* Prevent ugly wrapper from turning into dark box */
div[data-testid="stTextInput"] > div {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Label styling */
div[data-testid="stTextInput"] label {
    font-size: 0.95rem !important;
    color: #ffb86c !important;
    margin-bottom: 6px !important;
    background: none !important;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# 🚀 INITIALIZATION
# ======================================================
# Path του logo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "assets", "images", "logo.png")

# Μετατροπή εικόνας σε base64 για inline εμφάνιση
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
else:
    st.warning(f"⚠️ Logo not found at {logo_path}")

# Εμφάνιση inline logo + text
st.markdown(f"""
<div style="display:flex;align-items:center;gap:25px;margin-top:-10px;margin-bottom:20px;">
    <img src="data:image/png;base64,{logo_base64}" width="100" style="border-radius:10px;"/>
    <div>
        <h1 style="margin-bottom:0;">Content Search AI</h1>
        <p style="margin-top:4px;color:#9aa0a6;font-size:1.1rem;">
            Search Content in Multimedia Digital Archives using AI
        </p>
        <p style="margin-top:-8px;color:#9aa0a6;font-size:0.9rem;">Version 1.7</p>
    </div>
</div>
""", unsafe_allow_html=True)

DATA_DIR = "./data"
model = Model()
model.download_model()
searcher = ImageSearcher()
audio = AudioSearcher()

pdf = PDFSearcher()
# ======================================================
# 🧭 TABS SETUP
# ======================================================
tabs = st.tabs([
    "📊 Dashboard",
    "ℹ️ Application Info",
    "⚙️ Application Settings",
    "💬 Search: Text → Image",
    "🖼️ Search: Image → Image",
    "💬 Search: Text → PDF",
    "📚 Search: PDF → PDF",
    "🎧 Search: Text → Audio"
])

# ======================================================
# 📊 DASHBOARD
# ======================================================
with tabs[0]:
    st.subheader("📊 System Dashboard")

    # Manual refresh button (optional)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    # LIVE CPU / RAM
    cpu_percent = psutil.cpu_percent(interval=0.3)
    ram_percent = psutil.virtual_memory().percent

    st.markdown(f"""
    <div class="dashboard-grid">
        <div class="dash-card">
            <h3>🧠 System Overview</h3>
            <p><strong>CPU Usage:</strong> {cpu_percent}%</p>
            <p><strong>RAM Usage:</strong> {ram_percent}%</p>
        </div>

        <div class="dash-card">
            <h3>🖼 Images Watchdog</h3>
            <p>Placeholder – εδώ θα μπει status για επεξεργασία εικόνων.</p>
        </div>

        <div class="dash-card">
            <h3>📄 PDFs Watchdog</h3>
            <p>Placeholder – εδώ θα μπει status για PDF indexing.</p>
        </div>

        <div class="dash-card">
            <h3>🎧 Audio Watchdog</h3>
            <p>Placeholder – εδώ θα μπει status για audio & emotions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# ℹ️ APPLICATION INFORMATION TAB
# ======================================================
with tabs[1]:
    st.subheader("ℹ️ Application Information")

    # ======================================================
    # 🧠 ABOUT THIS PROJECT — CARD
    # ======================================================
    with st.container():
        with st.expander("🧠 About This Project", expanded=True):
            st.markdown("""
                This system is a **unified multimodal retrieval platform** capable of searching across  
                **Images, PDFs, Audio**, and **Text**, all inside a single shared semantic embedding space.

                It demonstrates practical and research-level techniques in:
                - **Image Search** (text → image, image → image)  
                - **PDF Document Search** (text → PDF, PDF → PDF)  
                - **Audio Semantic Search** (Whisper transcription + M-CLIP embeddings)  
                - **Emotion Detection** (Emotion Model V5)  
                - **Real-time indexing using Watchdogs**  

                A major milestone is the completion of **v1.7**, which replaces  
                all local embedding files with a **unified SQLite multimodal database**  
                and introduces **automatic real-time filesystem indexing**.

                ---
                ### 🧩 Technologies Used
                - **Python 3.11**
                - **Streamlit**
                - **PyTorch / Sentence-Transformers**
                - **OpenAI Whisper & Faster-Whisper**
                - **M-CLIP multilingual model**
                - **Emotion Model V5**
                - **PyMuPDF**
                - **FAISS**
                - **SQLite3**

                ---
                ### ⚙️ Model Architecture Overview
                - **M-CLIP (ViT-B/32)** → unified text/image/PDF/audio embeddings  
                - **Whisper-small** → speech-to-text transcription  
                - **Audio semantic encoder** → transcript embeddings with M-CLIP  
                - **Emotion Model V5** → 6-class emotional classification  
                - **Per-page PDF encoder** → M-CLIP page embeddings  

                Together, these components form a **cross-modal AI retrieval engine**  
                supporting fully multilingual queries.
            """)

    # ======================================================
    # 📘 VERSION HISTORY — CARD
    # ======================================================
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    with st.container():
        with st.expander("📘 Version History", expanded=False):
            st.markdown("""
                ## 🟢 **v1.7 — Full Multimodal SQLite Integration & Real-Time Indexing (November 2025)**  
                This is the largest structural update so far.  

                ### 🔥 Highlights
                - Introduced a **single unified SQLite database** for all modalities:
                    - `images`
                    - `pdf_pages`
                    - `audio_embeddings`
                    - `audio_emotions`
                - Removed **all old embedding/transcript folders**:
                    - `data/transcripts/`
                    - `data/embeddings/`
                    - `data/transcripts/embeds/`
                    - all `.npy` and `.txt` cache files  
                - Full **relative path normalization** for cross-platform compatibility  
                - Rebuilt **all transcripts** with Whisper  
                - Rebuilt **all emotion predictions** with the V5 model  
                - Introduced **Watchdog services** for:
                    - 🔄 Images  
                    - 📄 PDFs  
                    - 🎧 Audio  
                - Automatic:
                    - detection of file creation/deletion  
                    - embedding extraction  
                    - DB insertion/removal  
                - Removed all manual “rebuild” buttons from UI  
                - Massive codebase cleanup and folder restructuring  

                ---
                ## 🟢 **v1.6 — Audio Search Integration (November 2025)**
                - Whisper transcription engine  
                - M-CLIP semantic audio search  
                - Emotion Model V5 integration  
                - Word-level timestamp detection  
                - Audio visualization module  

                ---
                ## 🟢 **v1.5 — Stable Release (October 2025)**
                - Full PDF search module  
                - PDF per-page processing  
                - Document similarity module  
                - UI improvements  

                ---
                ## 🟠 **v1.4 — Core Integration (September 2025)**
                - Modular UI  
                - Caching system  
                - Layout refactoring  

                ---
                ## 🟡 **v1.3 — M-CLIP (August 2025)**
                - Multilingual CLIP  
                - Unified embedding space  

                ---
                ## 🔵 **v1.2 — Visual Search Prototype (June 2025)**
                - Text → Image  
                - Image → Image  

                ---
                ## ⚪ **v1.1 — Research Setup (May 2025)**  

                ---
                ## ⚫ **v1.0 — Project Start (April 2025)**
            """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# ⚙️ SETTINGS TAB WITH ACCORDIONS
# ======================================================
with tabs[2]:
    st.subheader("⚙️ Application Settings")
    # ------------------------------------------------------
    # DISPLAY SETTINGS
    # ------------------------------------------------------
    with st.expander("🔧 Display Settings", expanded=True):
        top_k = st.slider("Select number of results per search", 3, 30, 5)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 💬 TEXT → IMAGE SEARCH
# ======================================================
with tabs[3]:
    st.subheader("💬 Text-to-Image Search")

    # ----------------------------------
    # State init
    # ----------------------------------
    if "run_text_search" not in st.session_state:
        st.session_state.run_text_search = False

    def trigger_text_search():
        st.session_state.run_text_search = True

    # ----------------------------------
    # Input
    # ----------------------------------
    query = st.text_input(
        "✍️ Enter your search query",
        value="",
        on_change=trigger_text_search
    )

    if st.button("🔎 Run Text Search"):
        st.session_state.run_text_search = True

    # ----------------------------------
    # Run search
    # ----------------------------------
    if st.session_state.run_text_search:
        if not query.strip():
            st.warning("⚠️ Please enter a search phrase.")
        else:
            st.info(f"Searching for: '{query}' ...")

            results = searcher.search(query, top_k=top_k)

            if not results:
                st.warning("No results found.")
            else:
                cols = st.columns(len(results))

                for idx, r in enumerate(results):
                    score = r["score"]
                    confidence = r.get("confidence", None)

                    # -------------------------
                    # Explainability text
                    # -------------------------
                    explain_text = ""
                    if confidence is not None:
                        if confidence >= 0.7:
                            explain_text = "🟢 High semantic relevance"
                        elif confidence >= 0.4:
                            explain_text = "🟡 Partial semantic match"
                        else:
                            explain_text = "🔴 Low confidence – weak semantic overlap"

                    # -------------------------
                    # Caption
                    # -------------------------
                    caption = f"Similarity: {score * 100:.2f}%"
                    if confidence is not None:
                        caption += f"\nConfidence: {confidence * 100:.1f}%"
                        caption += f"\n{explain_text}"

                    cols[idx].image(
                        r["path"],
                        caption=caption,
                        use_container_width=True
                    )

    # ----------------------------------
    # Reset trigger
    # ----------------------------------
    st.session_state.run_text_search = False

# ======================================================
# 🖼️ IMAGE → IMAGE SEARCH
# ======================================================
with tabs[4]:
    st.subheader("🖼️ Image-to-Image Search")
    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        query_image_path = os.path.join("data/query_images", uploaded_file.name)
        os.makedirs(os.path.dirname(query_image_path), exist_ok=True)

        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(query_image_path, caption="📸 Uploaded Image", width=250)

        if st.button("🔍 Run Image Search"):
            st.info("Analyzing and comparing image...")
            start = time.time()
            results = searcher.search_by_image(query_image_path, top_k=top_k)
            elapsed = time.time() - start

            if not results:
                st.warning("No similar images found.")
            else:
                st.success(f"✅ Found {len(results)} similar images in {elapsed:.2f}s")

                cols = st.columns(len(results))  # ✅ FIX

                for idx, r in enumerate(results):
                    caption = f"Similarity: {r['score'] * 100:.2f}%"

                    if r.get("confidence") is not None:
                        caption += f"\nConfidence: {r['confidence'] * 100:.1f}%"

                    caption += "\nReason: visual embedding similarity"

                    cols[idx].image(
                        r["path"],
                        caption=caption,
                        use_container_width=True
                    )

# ======================================================
# 💬 TEXT → PDF SEARCH
# ======================================================
with tabs[5]:
    st.subheader("💬 Text-to-PDF Semantic Search")
    query_text = st.text_area("✍️ Enter your search text:", placeholder="e.g. deep learning in medical imaging")

    if st.button("🔍 Run Text → PDF Search"):
        if not query_text.strip():
            st.warning("⚠️ Please enter text before searching.")
        else:
            st.info(f"Searching for: '{query_text}' ...")

            searcher = PDFSearcher(db_path="content_search_ai.db")

            with st.spinner("Processing and comparing PDFs..."):
                results = searcher.search_by_text(query_text=query_text, top_k=top_k)

            if not results:
                st.warning("❌ No matching PDFs found.")
            else:
                st.success(f"✅ Found {len(results)} relevant PDFs!")

                for r in results:
                    filename = os.path.basename(r["pdf"])

                    st.markdown(
                        f"### 📄 {filename} (Page {r['page']}) — Score: `{r['score']:.4f}`"
                    )
                    st.caption(f"**Snippet:** {r['snippet']}")

                    with open(r["pdf"], "rb") as f:
                        pdf_data = f.read()

                    st.download_button(
                        label=f"⬇️ Download {filename}",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_{filename}_{r['page']}"
                    )
                    st.markdown("---")

# ======================================================
# 📚 PDF → PDF SEARCH
# ======================================================
with tabs[6]:
    st.subheader("📚 PDF-to-PDF Similarity Search")

    uploaded_pdf = st.file_uploader("📤 Upload a PDF to compare", type=["pdf"])
    base_folder = "./data/pdfs"
    query_folder = "./data/query"

    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(query_folder, exist_ok=True)

    if uploaded_pdf is not None:
        # Save uploaded PDF to query folder
        query_path = os.path.join(query_folder, uploaded_pdf.name)
        with open(query_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_pdf.name}")
        st.info("Analyzing document similarity...")

        # SQLite-powered PDF searcher
        searcher = PDFSearcher(db_path="content_search_ai.db")

        with st.spinner("Processing and comparing PDFs..."):
            results = searcher.search_similar_pdfs(
                query_pdf_path=query_path,
                top_k=top_k
            )

        if not results:
            st.warning("❌ No strong matches found.")
        else:
            st.success(f"✅ Found {len(results)} similar documents.")

            for r in results:
                filename = os.path.basename(r["pdf"])
                color = "🟢" if r["score"] >= 0.98 else "🟠" if r["score"] >= 0.95 else "🔴"

                st.markdown(
                    f"### {color} {filename} — Page {r['page']} — Score: `{r['score']:.4f}`"
                )
                st.caption(f"**Snippet:** {r['snippet']}")

                # Load PDF file bytes for download
                with open(r["pdf"], "rb") as f:
                    pdf_data = f.read()

                st.download_button(
                    label=f"⬇️ Download {filename}",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    key=f"download_{filename}_{r['page']}"
                )

                st.markdown("---")

# ======================================================
# 🎧 AUDIO SEARCH (PLACEHOLDER)
# ======================================================
with tabs[7]:
    st.subheader("🎧 Text-to-Audio Search (Semantic + Emotion + Language Filter)")

    with st.container():
        st.markdown("""
        #### 🎨 Color Guide
        - 🟧 **Orange:** Exact location where your query was detected in the audio  
        - 🎭 **Emotion background (soft color):** Detected overall emotion of the audio  
            - 😡 **Red** → Angry  
            - 🤢 **Purple (Dark)** → Disgust  
            - 😱 **Purple (Light)** → Fearful  
            - 😊 **Green** → Happy  
            - 😐 **Gray** → Neutral  
            - 😢 **Blue** → Sad  
        """)

    # -------------------------------
    # STATE για trigger από Enter
    # -------------------------------
    if "run_audio_search" not in st.session_state:
        st.session_state.run_audio_search = False

    def trigger_audio_search():
        st.session_state.run_audio_search = True

    # -------------------------------
    # TEXT INPUT (ENTER triggers search)
    # -------------------------------
    query = st.text_input(
        "🔎 Enter your audio search phrase",
        value="",
        on_change=trigger_audio_search
    )

    # -------------------------------
    # BUTTON (also triggers search)
    # -------------------------------
    run_btn = st.button("Run Audio Search", use_container_width=True)
    if run_btn:
        st.session_state.run_audio_search = True

    # -------------------------------
    # RUN SEARCH (button or Enter)
    # -------------------------------
    if st.session_state.run_audio_search:

        if not query.strip():
            st.warning("⚠️ Please enter a phrase.")
        else:
            with st.spinner("Searching audio…"):
                query_type = audio.classify_query_type(query)

                if query_type == "emotion":
                    results = audio.search_by_emotion(query, top_k=top_k)
                else:
                    results = audio.search_semantic_emotion(query, top_k=top_k)

        if not results:
            st.error("❌ No matching audio found.")
        else:
            st.success(f"✅ Found {len(results)} audio matches!")

            for r in results:
                fname = r["filename"]
                folder = r["folder"]
                semantic = r["similarity"]
                emotion = r.get("emotion", None)
                transcript = r.get("transcript", "")
                lang = r.get("text_language", "unknown")

                full_path = Path(r["full_path"]).as_posix()
                tools = CoreTools(full_path)

                st.markdown(f"""
                ### 🎵 {fname}
                **Folder:** `{folder}`  
                🌐 **Language:** `{lang}`  
                🔊 **Semantic Similarity:** `{semantic:.3f}`  
                🎭 **Emotion:** `{emotion}`
                """)

                # --- QUERY SEGMENTS HIGHLIGHT ---
                try:
                    segments = audio.get_query_segments(Path(full_path), query)
                except Exception as e:
                    segments = []
                    st.warning(f"Could not compute query segments: {e}")

                st.write("### 📊 Audio Visualization")

                tools.plot_waveform_and_spectrogram_with_highlights(
                    query_segments=segments, emotion_label=r["emotion"]
                )

                with st.expander("📄 Transcript"):
                    st.write(transcript)

                try:
                    with open(full_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                    st.caption(full_path)
                except Exception as e:
                    st.error(f"Could not load audio file `{full_path}`: {e}")

                st.markdown("---")

    # reset για να μην ξανατρέχει σε κάθε refresh
    st.session_state.run_audio_search = False

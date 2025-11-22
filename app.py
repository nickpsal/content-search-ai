import os
import time
import streamlit as st
import base64
from pathlib import Path
from core import ImageSearcher, PDFSearcher, Model, AudioSearcher, CoreTools

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
        <p style="margin-top:-8px;color:#9aa0a6;font-size:0.9rem;">Version 1.6</p>
    </div>
</div>
""", unsafe_allow_html=True)

DATA_DIR = "./data"
model = Model()
model.download_model()
searcher = ImageSearcher()
audio = AudioSearcher()

pdf = PDFSearcher()
pdf.download_pdf_data()

# ======================================================
# 🧭 TABS SETUP
# ======================================================
tabs = st.tabs([
    "ℹ️ Application Info",
    "⚙️ Application Settings",
    "💬 Search: Text → Image",
    "🖼️ Search: Image → Image",
    "📚 Search: PDF → PDF",
    "💬 Search: Text → PDF",
    "🎧 Search: Text → Audio",
    "🎥 Search: Video Search"
])

# ======================================================
# ⚙️ SETTINGS TAB WITH ACCORDIONS
# ======================================================
with tabs[1]:
    st.subheader("⚙️ Application Settings")

    # ------------------------------------------------------
    # DATASET & EMBEDDINGS CONFIG
    # ------------------------------------------------------
    with st.expander("⚙️ Dataset & Embeddings Configuration", expanded=True):

        st.markdown("### 🖼️ Image Processing")
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            if st.button("📦 Download COCO Dataset", use_container_width=True):
                with st.spinner("Downloading COCO Dataset..."):
                    searcher.download_coco_data()
                st.success("✅ COCO dataset downloaded successfully!")

        with col2:
            if st.button("🧠 Extract Image Embeddings", use_container_width=True):
                with st.spinner("Extracting Image Embeddings..."):
                    searcher.extract_image_embeddings()
                    searcher.extract_image_embeddings('other')
                st.success("✅ Image embeddings created successfully!")

        with col3:
            if st.button("💬 Extract Caption Embeddings", use_container_width=True):
                with st.spinner("Extracting Caption Embeddings..."):
                    searcher.extract_text_embeddings()
                st.success("✅ Caption embeddings created successfully!")

        # --------------------------------------------------
        # AUDIO
        # --------------------------------------------------
        st.markdown("### 🎧 Audio Processing")
        col4, col5, col6 = st.columns([1, 1, 1], gap="medium")

        with col4:
            if st.button("🎙️ Build Audio Embeddings + Transcripts", use_container_width=True):
                with st.spinner("Transcribing audio and building embeddings..."):
                    audio.build_all_transcripts()
                st.success("✅ Audio transcripts and embeddings created successfully!")

        with col5:
            if st.button("🎭 Build Emotion Cache File", use_container_width=True):
                with st.spinner("Creating emotion cache file..."):
                    audio.save_emotion_cache()
                st.success("✅ Emotion cache file created successfully!")

    # ------------------------------------------------------
    # DISPLAY SETTINGS
    # ------------------------------------------------------
    with st.expander("🔧 Display Settings", expanded=False):
        top_k = st.slider("Select number of results per search", 3, 30, 5)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# ℹ️ APPLICATION INFORMATION TAB
# ======================================================
with tabs[0]:
    st.subheader("ℹ️ Application Information")

    # ======================================================
    # 🧠 ABOUT THIS PROJECT — CARD
    # ======================================================
    with st.container():
        with st.expander("🧠 About This Project", expanded=True):
            st.markdown("""
                This system is a **unified multimodal retrieval platform** capable of searching across  
                **images, text, PDFs, and audio**, using a shared semantic embedding space.

                It demonstrates practical and research-level techniques in:
                - **Image Search** (text → image, image → image)  
                - **PDF Document Search** (text → PDF, PDF → PDF)  
                - **Audio Semantic Search** (text → audio using Whisper + projection)

                A major new milestone is the completion of the **Audio-Align v2 Emotion Model (v5)**,  
                which aligns Whisper audio embeddings with the M-CLIP text/image embedding space  
                enabling **high-precision audio semantic retrieval** and **emotion-based audio search**.

                ---
                ### 🧩 Technologies Used
                - **Python 3.11**
                - **Streamlit**
                - **PyTorch**
                - **Sentence-Transformers**  
                - **OpenAI Whisper**  
                - **PyMuPDF**, **FFmpeg**, **TQDM**, **PIL**, **NumPy**

                ---
                ### ⚙️ Model Architecture Summary
                - **M-CLIP (ViT-B/32)**  
                - **Whisper-small encoder**  
                - **Audio Projection Layer (512-D)**  
                - **Emotion Classifier (6 classes)**  
                - **PDF Encoder**

                Combined, these models enable **cross-modal semantic retrieval**  
                across previously unrelated media types.
            """)

    # ======================================================
    # 📘 VERSION HISTORY — CARD
    # ======================================================
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    with st.container():
        with st.expander("📘 Version History", expanded=False):
            st.markdown("""
                ## 🟢 **v1.6 — Audio Search Integration (November 2025)**  
                - Integrated **Audio Semantic Search**  
                - Added dual audio folders  
                - Whisper transcription engine  
                - Hybrid search  
                - Cached audio embeddings  
                - Preprocessing pipeline  

                ---
                ## 🟢 **v1.5 — Stable Release (October 2025)**
                - Full **PDF-to-PDF** and **Text-to-PDF** retrieval  
                - Added Application Info tab  
                - Improved UI design  
                - Refined thresholds

                ---
                ## 🟠 **v1.4 — Core Integration (September 2025)**
                - Modular UI tabs  
                - Global caching system  

                ---
                ## 🟡 **v1.3 — M-CLIP Integration (August 2025)**
                - Multilingual CLIP  
                - Cross-modal semantic search  

                ---
                ## 🔵 **v1.2 — Visual Search Prototype (June 2025)**
                - Text → Image  
                - Image → Image  
                - COCO experiments  

                ---
                ## ⚪ **v1.1 — Research Setup (May 2025)**
                - Dataset initialization  

                ---
                ## ⚫ **v1.0 — Project Start (April 2025)**
                - Research planning
            """)
    st.markdown('</div>', unsafe_allow_html=True)

    # ======================================================
    # 🧾 NEXT PLANNED UPDATES — CARD
    # ======================================================
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    with st.container():
        with st.expander("🧾 Next Planned Updates", expanded=False):
            st.markdown("""
                - 🎥 Video search (frame-level embeddings)  
                - 🎚️ Hybrid audio-video retrieval  
                - 🗂️ Metadata-based ranking  
                - 📊 Heatmap visualization  
                - ⚡ Faster keyword segmentation  
            """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# 💬 TEXT → IMAGE SEARCH
# ======================================================
with tabs[2]:
    st.subheader("💬 Text-to-Image Search")

    # state για να κάνουμε trigger το search
    if "run_text_search" not in st.session_state:
        st.session_state.run_text_search = False

    def trigger_text_search():
        st.session_state.run_text_search = True

    query = st.text_input(
        "✍️ Enter your search query",
        value="",
        on_change=trigger_text_search  # <-- ENTER triggers search
    )

    run_btn = st.button("🔎 Run Text Search")

    # Run search if: Enter pressed OR button clicked
    if run_btn:
        st.session_state.run_text_search = True

    if st.session_state.run_text_search:
        if not query.strip():
            st.warning("⚠️ Please enter a search phrase.")
        else:
            st.info(f"Searching for: '{query}' ...")
            start = time.time()
            results = searcher.search(query, top_k=top_k)
            elapsed = time.time() - start

            if results:
                cols = st.columns(top_k)
                for idx, r in enumerate(results[:top_k]):
                    img_path = r["path"]
                    score = r["score"]
                    source = "COCO" if "val2017" in img_path else "Other"

                    cols[idx].image(
                        img_path,
                        caption=f"Similarity: {score * 100:.2f}% | Dataset: {source}",
                        use_container_width=True
                    )

    # reset state after rendering
    st.session_state.run_text_search = False

# ======================================================
# 🖼️ IMAGE → IMAGE SEARCH
# ======================================================
with tabs[3]:
    st.subheader("🖼️ Image-to-Image Search")
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

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
                cols = st.columns(top_k)
                for idx, r in enumerate(results[:top_k]):
                    img_path = r["path"]
                    score = r["score"]
                    source = "COCO" if "val2017" in img_path else "Other"

                    cols[idx].image(
                        img_path,
                        caption=f"Similarity: {score * 100:.2f}% | Dataset: {source}",
                        use_container_width=True
                    )

# ======================================================
# 📚 PDF → PDF SEARCH
# ======================================================
with tabs[4]:
    st.subheader("📚 PDF-to-PDF Similarity Search")

    uploaded_pdf = st.file_uploader("📤 Upload a PDF to compare", type=["pdf"])
    base_folder = "./data/pdfs"
    query_folder = "./data/query"
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(query_folder, exist_ok=True)

    if uploaded_pdf is not None:
        query_path = os.path.join(query_folder, uploaded_pdf.name)
        with open(query_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_pdf.name}")
        st.info("Analyzing document similarity...")

        searcher = PDFSearcher("./models/mclip_finetuned_coco_ready")

        with st.spinner("Processing and comparing PDFs..."):
            results = searcher.search_similar_pdfs(query_pdf=query_path, folder=base_folder, top_k=top_k)

        if not results:
            st.warning("❌ No strong matches found.")
        else:
            st.success(f"✅ Found {len(results)} similar documents.")
            for r in results:
                color = "🟢" if r["score"] >= 0.98 else "🟠" if r["score"] >= 0.95 else "🔴"
                st.markdown(f"### {color} {r['file']} — Page {r['page']} — Score: `{r['score']:.4f}`")
                st.caption(f"**Snippet:** {r['snippet']}")
                pdf_path = os.path.join(base_folder, r["file"])
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label=f"⬇️ Download {r['file']}",
                    data=pdf_data,
                    file_name=r["file"],
                    mime="application/pdf"
                )
                st.markdown("---")

# ======================================================
# 💬 TEXT → PDF SEARCH
# ======================================================
with tabs[5]:
    st.subheader("💬 Text-to-PDF Semantic Search")
    query_text = st.text_area("✍️ Enter your search text:", placeholder="e.g. deep learning in medical imaging")

    base_folder = "./data/pdfs"
    os.makedirs(base_folder, exist_ok=True)

    if st.button("🔍 Run Text → PDF Search"):
        if not query_text.strip():
            st.warning("⚠️ Please enter text before searching.")
        else:
            st.info(f"Searching for: '{query_text}' ...")

            searcher = PDFSearcher("./models/mclip_finetuned_coco_ready")

            with st.spinner("Analyzing PDFs..."):
                results = searcher.search_by_text(query_text, folder=base_folder, top_k=top_k)

            if not results:
                st.warning("No matching PDFs found.")
            else:
                st.success(f"✅ Found {len(results)} relevant PDFs!")
                for r in results:
                    st.markdown(f"### 📄 {r['file']} (Page {r['page']}) — Score: `{r['score']:.4f}`")
                    st.caption(f"**Snippet:** {r['snippet']}")
                    pdf_path = os.path.join(base_folder, r["file"])
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label=f"⬇️ Download {r['file']}",
                        data=pdf_data,
                        file_name=r["file"],
                        mime="application/pdf",
                        key=f"download_{r['file']}_{r['page']}"
                    )

# ======================================================
# 🎧 AUDIO SEARCH (PLACEHOLDER)
# ======================================================
with tabs[6]:
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

# ======================================================
# 🎥 VIDEO SEARCH (PLACEHOLDER)
# ======================================================
with tabs[7]:
    st.subheader("🎥 Video Search (Coming Soon)")
    st.info("Video similarity search will be implemented in a future version.")

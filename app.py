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
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 25px;
}
.result-card {
    position: relative;
    background-color: #1e1e1e;
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
}
.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 20px rgba(255,255,255,0.2);
}
.result-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 14px;
}
.overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.9) 100%);
    color: white;
    padding: 10px;
    text-align: center;
}
.score-label {
    color: #ff6b6b;
    font-weight: 700;
    font-size: 0.9rem;
}
.source-label {
    color: #bbb;
    font-size: 0.8rem;
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
searcher = ImageSearcher(data_dir=DATA_DIR)
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
    with st.expander("⚙️ Dataset & Embeddings Configuration", expanded=False):
        st.markdown("### 🎧 Image Processing")
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            if st.button("📦 Download COCO Dataset", use_container_width=True):
                searcher.download_coco_data()
                st.success("✅ COCO dataset downloaded successfully!")

        with col2:
            if st.button("🧠 Extract Image Embeddings", use_container_width=True):
                searcher.extract_image_embeddings()
                st.success("✅ Image embeddings created successfully!")

        with col3:
            if st.button("💬 Extract Caption Embeddings", use_container_width=True):
                searcher.extract_text_embeddings()
                st.success("✅ Caption embeddings created successfully!")

        # ---------------------------------------------
        # NEW ROW — AUDIO
        # ---------------------------------------------
        st.markdown("### 🎧 Audio Processing")

        a1, a2, _ = st.columns([1, 1, 1], gap="medium")

        with a1:
            if st.button("🎵 Build Audio Embeddings", use_container_width=True):
                with st.spinner("Building audio embeddings…"):
                    audio.build_all_embeddings()
                st.success("✅ Audio embeddings built!")

        with a2:
            if st.button("📝 Build Audio Transcripts", use_container_width=True):
                with st.spinner("Transcribing audio…"):
                    audio.build_all_transcripts()
                st.success("✅ Audio transcripts created!")

    # ------------------------------------------------------
    # DISPLAY SETTINGS
    # ------------------------------------------------------
    with st.expander("🔧 Display Settings", expanded=False):
        top_k = st.slider("Select number of results per search", 3, 30, 5)

# ======================================================
# ℹ️ APP INFO TAB
# ======================================================
with tabs[0]:
    st.subheader("ℹ️ Application Information")

    # ===========================
    # ABOUT THE PROJECT
    # ===========================
    with st.expander("🧠 About This Project", expanded=True):
        st.markdown("""
            This system is a **unified multimodal retrieval platform** capable of searching across  
            **images, text, PDFs, and audio**, using a shared semantic embedding space.

            It demonstrates practical and research-level techniques in:
            - **Image Search** (text → image, image → image)  
            - **PDF Document Search** (text → PDF, PDF → PDF)  
            - **Audio Semantic Search** (text → audio using Whisper + projection)  

            A major new milestone is the completion of the **Audio-Align v2 Emotion Model (v4)**,  
            which aligns Whisper audio embeddings with the M-CLIP text/image embedding space  
            for **high-precision audio semantic retrieval**.

            ---
            ### 🧩 Technologies Used
            - **Python 3.11**
            - **Streamlit** — interactive UI  
            - **PyTorch** — deep learning backend  
            - **Sentence-Transformers** — Multilingual CLIP  
            - **OpenAI Whisper** (fine-tuned + projection)  
            - **PyMuPDF** — PDF parsing  
            - **FFmpeg, TQDM, PIL, NumPy** — preprocessing utilities  

            ---
            ### ⚙️ Model Architecture Summary
            - **M-CLIP (ViT-B/32)** — multilingual text & image embeddings  
            - **Whisper-small encoder** — audio feature extraction  
            - **Audio Projection Layer (512-D)** — trained to align audio with CLIP space  
            - **Emotion Classification Head (6 classes)** — trained on RAVDESS/CREMA-D  
            - **PDF encoder** — semantic page-level representations  

            The combination of these models enables **cross-modal semantic retrieval**  
            across previously unrelated media types.

            ---
            ### 👨‍💻 Developer
            **Nikolaos Psaltakis**  
            University of West Attica  
            Department of Informatics & Computer Engineering  
            Bachelor Thesis Project – © 2025
        """)

    # ===========================
    # VERSION HISTORY
    # ===========================
    with st.expander("📘 Version History", expanded=False):
        st.markdown("""
            ## 🟢 **v1.6 – Audio Search Integration (November 2025)**  
            - Added **Audio Semantic Search module** using Whisper + Projection  
            - Implemented **AudioSearcher class** (embeddings, transcripts, hybrid search)  
            - Added **dual-folder audio support** (AudioWAV + other_audio)  
            - Added **Whisper transcription engine** for audio-to-text retrieval  
            - Introduced **Hybrid Search** combining audio embeddings + transcripts  
            - Enabled **fast cached embeddings** for immediate reloading  
            - Streamlined dataset preprocessing and environment cleanup  
            - Prepared for full multimodal demonstration in Streamlit UI  

            ---
            ### 🟢 **v1.5 – Stable Release (October 2025)**
            - Added **PDF-to-PDF** & **Text-to-PDF** semantic search  
            - Introduced **App Info tab** with detailed metadata  
            - Improved Streamlit UI, multilingual support & documentation  
            - Cleaned hybrid CLIP + M-CLIP pipeline  
            - Refined similarity thresholds and result ranking  

            ### 🟠 **v1.4 – Core Functionality Integration (September 2025)**
            - Modular UI with Streamlit tabs  
            - Stable caching of all embeddings  
            - Added embedded settings & controls  

            ### 🟡 **v1.3 – Multilingual CLIP Integration (August 2025)**
            - M-CLIP integration with Greek + English support  
            - Added cross-modal retrieval foundation  
            - Initial PDF search engine implementation  

            ### 🔵 **v1.2 – Visual Search Prototype (June 2025)**
            - Text-to-image & image-to-image CLIP search  
            - COCO dataset evaluation  
            - Initial embedding store format  

            ### ⚪ **v1.1 – Research Setup (May 2025)**
            - Environment setup, dataset initialization  
            - First preprocessing & validation tools  

            ### ⚫ **v1.0 – Project Initialization (April 2025)**
            - Thesis planning & architecture specification  
        """)

    with st.expander("🧾 Next Planned Updates", expanded=False):
        st.markdown("""
            - 🎥 Integrate **video search** using frame-level M-CLIP embeddings  
            - 🎚️ Add **hybrid audio-video retrieval**  
            - 🗂️ Introduce metadata-based ranking (speaker, emotion, duration)  
            - 📊 Analytics panel for embedding similarity visualization  
        """)

# ======================================================
# 💬 TEXT → IMAGE SEARCH
# ======================================================
with tabs[2]:
    st.subheader("💬 Text-to-Image Search")
    query = st.text_input("✍️ Enter your search query")

    if st.button("🔎 Run Text Search"):
        if not query.strip():
            st.warning("⚠️ Please enter a search phrase.")
        else:
            st.info(f"Searching for: '{query}' ...")
            start = time.time()
            results = searcher.search(query, top_k=top_k, verbose=False)
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

    query = st.text_input("🔎 Enter your audio search phrase")

    if st.button("Run Audio Search", use_container_width=True):
        if not query.strip():
            st.warning("⚠️ Please enter a phrase.")
        else:
            with st.spinner("Searching audio…"):
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

                    #f"[{i}] {r['filename']}  ({r['folder']})"
                    # Convert Windows path → POSIX
                    full_path = Path(r["full_path"]).as_posix()

                    tools = CoreTools(full_path)

                    st.markdown(f"""
                    ### 🎵 {fname}
                    **Folder:** `{folder}`  
                    🌐 **Language:** `{lang}`  
                    🔊 **Semantic Similarity:** `{semantic:.3f}`  
                    🎭 **Emotion:** `{emotion}`
                    """)
                    tools.plot_waveform_and_spectrogram()

                    with st.expander("📄 Transcript"):
                        st.write(transcript)

                    # === AUDIO PLAYER ===
                    try:
                        with open(full_path, "rb") as f:
                            st.audio(f.read(), format="audio/wav")
                        st.caption(full_path)
                    except Exception as e:
                        st.error(f"Could not load audio file `{full_path}`: {e}")

                    st.markdown("---")

# ======================================================
# 🎥 VIDEO SEARCH (PLACEHOLDER)
# ======================================================
with tabs[7]:
    st.subheader("🎥 Video Search (Coming Soon)")
    st.info("Video similarity search will be implemented in a future version.")

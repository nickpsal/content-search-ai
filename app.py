import os
import time
import streamlit as st
from core import ImageSearcher, PDFSearcher, Model

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
st.title("🔎 Search Content in Multimedia Digital Archives using AI")
st.markdown("Version **1.5**")

DATA_DIR = "./data"
model = Model()
model.download_model()
searcher = ImageSearcher(data_dir=DATA_DIR)

pdf = PDFSearcher()
pdf.download_pdf_data()

# ======================================================
# 🧭 TABS SETUP
# ======================================================
tabs = st.tabs([
    "⚙️ Settings",
    "ℹ️ App Info",
    "💬 Text → Image",
    "🖼️ Image → Image",
    "📚 PDF → PDF",
    "💬 Text → PDF",
    "🎧 Audio Search",
    "🎥 Video Search"
])

# ======================================================
# ⚙️ SETTINGS TAB
# ======================================================
with tabs[0]:
    st.subheader("⚙️ Dataset & Embeddings Configuration")

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

    st.divider()
    st.subheader("🔧 Display Settings")
    top_k = st.slider("Select number of results per search", 3, 30, 5)
    st.info(f"Currently set to show up to {top_k} results per query.")

# ======================================================
# ℹ️ APP INFO TAB
# ======================================================
with tabs[1]:
    st.subheader("ℹ️ Application Information")

    st.markdown("""
    ### 🧠 About This Project
    This system demonstrates **content-based retrieval** across multiple media types:
    - **Images** — via multilingual CLIP embeddings (text-to-image & image-to-image)
    - **PDF Documents** — using semantic page-level similarity
    - **Audio & Video** — planned future extensions (Whisper & visual embedding extraction)

    ### 🧩 Technologies Used
    - **Python 3.10**
    - **Streamlit** for the interactive user interface
    - **PyTorch** and **Sentence-Transformers (M-CLIP)**
    - **OpenAI CLIP** for visual representation learning
    - **PyMuPDF** for text extraction from PDFs
    - **TQDM**, **PIL**, and **NumPy** for utilities and preprocessing

    ### ⚙️ Model Details
    The system employs a **fine-tuned Multilingual CLIP (ViT-B/32)** model  
    trained on the **COCO dataset** for robust multilingual text-image retrieval.

    ### 👩‍💻 Developer
    **Nikolaos Psaltakis**  
    University of West Attica  
    Department of Informatics and Computer Engineering  
    Bachelor Thesis Project – (c) 2025

    ---
    """)

    st.subheader("📘 Version History")

    st.markdown("""
    #### 🟢 **v1.5 – Stable Release (October 2025)**
    - Added **PDF-to-PDF** and **Text-to-PDF** semantic search  
    - Added **App Info tab** with About, Technologies, and Version History sections  
    - Improved **Streamlit UI design** and English and Greek documentation  
    - Refined **PDF similarity filtering** for cleaner results  
    - Updated **hybrid CLIP + M-CLIP pipeline**  
    - General code cleanup across `core/` modules  

    #### 🟠 **v1.4 – Core Functionality Integration (September 2025)**
    - Integrated **Streamlit tabs** for modular UI  
    - Optimized embeddings extraction and caching  
    - Added Settings tab for dataset and embedding control  

    #### 🟡 **v1.3 – Multilingual CLIP Implementation (August 2025)**
    - Integrated **M-CLIP (multilingual CLIP)** fine-tuning  
    - Added **cross-modal retrieval** for English and Greek queries  
    - Introduced initial PDF document similarity module  

    #### 🔵 **v1.2 – Visual Search Prototype (June 2025)**
    - Implemented **text-to-image** and **image-to-image** retrieval  
    - Added COCO dataset integration  
    - Established embedding storage and search indexing  

    #### ⚪ **v1.1 – Initial Research Setup (May 2025)**
    - Set up development environment  
    - Implemented model loading and preprocessing pipelines  
    - Built baseline retrieval testing framework  

    #### ⚫ **v1.0 – Project Initialization (April 2025)**
    - Defined thesis objectives and dataset structure  
    - Started architecture planning and repository setup  
    """)

    st.markdown("---")
    st.markdown("""
    🧾 **Next Planned Updates**
    - 🎧 Integrate **Whisper** for audio-to-text retrieval  
    - 🎥 Add **video similarity search** using frame-level embeddings  
    - ☁️ Enable **model caching and web deployment** on Streamlit Cloud  
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
    st.subheader("🎧 Audio Search (Coming Soon)")
    st.info("Audio similarity search will be implemented in the next release.")

# ======================================================
# 🎥 VIDEO SEARCH (PLACEHOLDER)
# ======================================================
with tabs[7]:
    st.subheader("🎥 Video Search (Coming Soon)")
    st.info("Video similarity search will be implemented in a future version.")

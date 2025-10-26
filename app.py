from core import ImageSearcher
from core import Model
import streamlit as st
from PIL import Image
import os
import time
import base64

# ======================================================
# 🌙 STYLING
# ======================================================
st.set_page_config(
    page_title="Search Content in Multimedia Digital Archives using Artificial Intelligence",
    layout="wide"
)

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
# 🧠 INIT
# ======================================================
st.title("🔎 Search Content in Multimedia Digital Archives using Artificial Intelligence")
st.markdown("Version **1.4**")

DATA_DIR = "./data"
model = Model()
model.download_model()
searcher = ImageSearcher(data_dir=DATA_DIR)

# ======================================================
# 🗂️ TABS
# ======================================================
tabs = st.tabs(["⚙️ Settings", "💬 Text Search", "🖼️ Image Search", "🎧 Audio Search", "🎥 Video Search"])

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
# 💬 TEXT → IMAGE SEARCH
# ======================================================
with tabs[1]:
    st.subheader("💬 Text-to-Image Search")
    query = st.text_input("✍️ Enter your search query")

    if st.button("🔎 Run Text Search"):
        if query.strip() == "":
            st.warning("⚠️ Please enter a search phrase.")
        else:
            st.info(f"Searching for: '{query}' ...")
            start = time.time()
            results = searcher.search(query, top_k=top_k, verbose=False)
            elapsed = time.time() - start

            if not results:
                st.warning("No results found.")
            else:
                st.success(f"✅ Found {len(results)} results in {elapsed:.2f}s")

                cards = []
                for r in results:
                    img_path = r["path"]
                    score = r["score"]
                    source = "COCO" if "val2017" in img_path else "Other"

                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    b64 = base64.b64encode(img_bytes).decode()

                    cards.append(f"""
                    <div class='result-card'>
                        <img src="data:image/jpeg;base64,{b64}" alt="{os.path.basename(img_path)}"/>
                        <div class="overlay">
                            <div class='score-label'>Similarity: {score*100:.2f}%</div>
                            <div class='source-label'>Dataset: {source}</div>
                        </div>
                    </div>
                    """)

                html = "<div class='result-grid'>" + "".join(cards) + "</div>"
                st.markdown(html, unsafe_allow_html=True)

# ======================================================
# 🖼️ IMAGE → IMAGE SEARCH
# ======================================================
with tabs[2]:
    st.subheader("🖼️ Image-to-Image Search")

    uploaded_file = st.file_uploader("📤 Upload an image to find similar ones", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        query_image_path = os.path.join("data/query_images", uploaded_file.name)
        os.makedirs(os.path.dirname(query_image_path), exist_ok=True)

        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(query_image_path, caption="📸 Uploaded Image", width=250)

        if st.button("🔍 Run Image Search"):
            st.info("Analyzing image and finding similar ones...")
            start = time.time()
            results = searcher.search_by_image(query_image_path, top_k=top_k, verbose=False)
            elapsed = time.time() - start

            if not results:
                st.warning("No similar images found.")
            else:
                st.success(f"✅ Found {len(results)} similar images in {elapsed:.2f}s")

                cards = []
                for r in results:
                    img_path = r["path"]
                    score = r["score"]
                    source = "COCO" if "val2017" in img_path else "Other"

                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    b64 = base64.b64encode(img_bytes).decode()

                    cards.append(f"""
                    <div class='result-card'>
                        <img src="data:image/jpeg;base64,{b64}" alt="{os.path.basename(img_path)}"/>
                        <div class="overlay">
                            <div class='score-label'>Similarity: {score*100:.2f}%</div>
                            <div class='source-label'>Dataset: {source}</div>
                        </div>
                    </div>
                    """)

                html = "<div class='result-grid'>" + "".join(cards) + "</div>"
                st.markdown(html, unsafe_allow_html=True)

# ======================================================
# 🎧 AUDIO SEARCH (Whisper placeholder)
# ======================================================
with tabs[3]:
    st.subheader("🎥 Audio Search (Future Feature)")
    st.info("Audio similarity search will be added in the next version.")

# ======================================================
# 🎥 VIDEO SEARCH (Future)
# ======================================================
with tabs[4]:
    st.subheader("🎥 Video Search (Future Feature)")
    st.info("Video similarity search will be added in the next version.")

from core import ImageSearcher
import streamlit as st
from deep_translator import GoogleTranslator
import os

# ---- ΡΥΘΜΙΣΕΙΣ ----
st.set_page_config(page_title="Search Content in Multimedia Digital Archives using Artificial Intelligence", layout="wide")
st.title("🔎 Search Content in Multimedia Digital Archives using Artificial Intelligence \n Version 1.0")

# ---- ΑΡΧΙΚΟΠΟΙΗΣΗ ----
DATA_DIR = "./data"
searcher = ImageSearcher(data_dir=DATA_DIR)

# ---- 1️⃣ DOWNLOAD COCO DATA ----
if st.button("📦 Download COCO Dataset"):
    searcher.download_coco_data()
    st.success("✅ The COCO dataset downloaded and unzipped successfully!")

# ---- 2️⃣ EXTRACT IMAGE EMBEDDINGS ----
if st.button("🧠 Extract Image Embeddings"):
    searcher.extract_image_embeddings()
    st.success("✅ Image embeddings was created successfully")

# ---- 3️⃣ EXTRACT TEXT EMBEDDINGS ----
if st.button("💬 Extract Caption Embeddings"):
    searcher.extract_text_embeddings()
    st.success("✅ Caption embeddings was created successfully!")

# ---- 4️⃣ SEARCH ----
st.divider()
st.subheader("🔍 Image Search")

query = st.text_input("✍️ Search Query")

if st.button("🔎 Run Search"):
    if query.strip() == "":
        st.warning("⚠️ You have to include a Search Criteria!")
    else:
        # Μετάφραση στα αγγλικά για το CLIP
        query_en = GoogleTranslator(source="auto", target="en").translate(query)
        st.info(f"Searching for: '{query}'")

        results = searcher.search(query_en, top_k=5)
        if not results:
            st.warning("No result Found.")
        else:
            st.success(f"Found {len(results)} results")

            cols = st.columns(5)
            for i, result in enumerate(results):
                img_path = result["path"]
                score = result["score"]

                with cols[i]:
                    st.image(img_path, caption=f"{os.path.basename(img_path)}", width=200)
                    st.progress(min(score, 1.0))  # progress bar μέχρι 1.0
                    st.caption(f"Similarity: **{score * 100:.2f}%**")

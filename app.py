from core import ImageSearcher
import streamlit as st
from deep_translator import GoogleTranslator
import os

# ---- ΡΥΘΜΙΣΕΙΣ ----
st.set_page_config(page_title="AI Image Search", layout="wide")
st.title("🔎 Αναζήτηση Εικόνων με CLIP & COCO Dataset")

# ---- ΑΡΧΙΚΟΠΟΙΗΣΗ ----
DATA_DIR = "./data"
searcher = ImageSearcher(data_dir=DATA_DIR)

# ---- 1️⃣ DOWNLOAD COCO DATA ----
if st.button("📦 Κατέβασε COCO Dataset"):
    searcher.download_coco_data()
    st.success("✅ Το COCO dataset κατέβηκε και αποσυμπιέστηκε επιτυχώς!")

# ---- 2️⃣ EXTRACT IMAGE EMBEDDINGS ----
if st.button("🧠 Δημιούργησε Image Embeddings"):
    searcher.extract_image_embeddings()
    st.success("✅ Δημιουργήθηκαν τα image embeddings!")

# ---- 3️⃣ EXTRACT TEXT EMBEDDINGS ----
if st.button("💬 Δημιούργησε Caption Embeddings"):
    searcher.extract_text_embeddings()
    st.success("✅ Δημιουργήθηκαν τα caption embeddings!")

# ---- 4️⃣ SEARCH ----
st.divider()
st.subheader("🔍 Αναζήτηση")

query = st.text_input("✍️ Γράψε ερώτημα (π.χ. 'Ένα άλογο στην παραλία')")

if st.button("🔎 Εκτέλεση Αναζήτησης"):
    if query.strip() == "":
        st.warning("⚠️ Πρέπει να γράψεις κάποιο ερώτημα!")
    else:
        # Μετάφραση στα αγγλικά για το CLIP
        query_en = GoogleTranslator(source="auto", target="en").translate(query)
        st.info(f"Αναζήτηση για: '{query}' → Μετάφραση: '{query_en}'")

        results = searcher.search(query_en, top_k=5)
        if not results:
            st.warning("Δεν βρέθηκαν αποτελέσματα.")
        else:
            st.success(f"Βρέθηκαν {len(results)} σχετικά αποτελέσματα:")

            cols = st.columns(5)
            for i, result in enumerate(results):
                img_path = result["path"]
                score = result["score"]

                with cols[i]:
                    st.image(img_path, caption=f"{os.path.basename(img_path)}", width=200)
                    st.progress(min(score, 1.0))  # progress bar μέχρι 1.0
                    st.caption(f"Similarity: **{score * 100:.2f}%**")

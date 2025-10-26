import os
import re
import fitz  # PyMuPDF
import torch
from tqdm import tqdm
import gdown
import zipfile
from sentence_transformers import SentenceTransformer, util

def word_overlap(a, b):
    """Calculate the number of shared words between two text segments."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return len(a_words.intersection(b_words))

class PDFSearcher:
    """
    PDF Similarity Searcher using a fine-tuned M-CLIP model.
    Works only with machine-readable PDFs (not scanned images).
    Performs per-page semantic comparison for higher precision.
    """
    def __init__(self, model_path="./models/mclip_finetuned_coco_ready", device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Loading M-CLIP model from: {model_path} on {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)
        self.min_score = 0.90  # minimum similarity threshold
        self.min_chars = 80    # minimum number of characters per page

    # ==========================================================
    # Download data pdf folder
    # ==========================================================
    @staticmethod
    def download_pdf_data():
        data_id = "1C4svye202Tt0cwK-tXY39QBQpyMioW_k"
        os.makedirs("./data", exist_ok=True)
        url = f"https://drive.google.com/uc?id={data_id}"
        output_path = "./data/pdf_data.zip"
        data_zip = os.path.join("./data", "pdf_data.zip")
        pdf_dir = os.path.abspath("./data")

        if len(os.listdir(pdf_dir)) > 0:
            print(f"✅ Model already exists in {pdf_dir}")
            return

        #Download Pdfs Data Folder from Google Drive
        print("\n📥 Downloading model from Google Drive...")
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        print(f"✅ File saved to: {output_path}")

        # Checking if it is a valid zip
        if not zipfile.is_zipfile(data_zip):
            print("❌ Downloaded file is not a valid ZIP. Check the Google Drive link ID.")
            return

        # eExtracting zip
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            files = zip_ref.namelist()
            print(f"\n📦 Extracting {len(files)} files into {pdf_dir}...")
            for file in tqdm(files, desc="Extracting", unit="file"):
                zip_ref.extract(file, pdf_dir)

        os.remove(data_zip)
        print(f"✅ Model extracted successfully into {data_zip}")

    # ==========================================================
    # Extract page embeddings (no OCR)
    # ==========================================================
    def get_pdf_pages_embeddings(self, pdf_path):
        pages = []
        valid_pages = 0
        total_pages = 0

        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)

                for i, page in enumerate(doc):
                    page_text = page.get_text("text").strip()
                    if not page_text or len(page_text) < self.min_chars:
                        continue

                    # Clean up unwanted characters and noise
                    clean_text = re.sub(r"[^A-Za-zΑ-Ωα-ω\s]", " ", page_text)
                    words = [w for w in clean_text.split() if len(w) > 2]
                    if len(words) < 10:
                        continue

                    # Text quality ratio (letters / total characters)
                    letters = sum(c.isalpha() for c in page_text)
                    ratio = letters / (len(page_text) + 1)
                    if ratio < 0.6:
                        continue

                    valid_pages += 1
                    emb = self.model.encode(page_text, convert_to_tensor=True, normalize_embeddings=True)
                    pages.append((i + 1, emb, page_text))

            # Skip documents with low text density
            if total_pages > 0:
                valid_ratio = valid_pages / total_pages
                if valid_ratio < 0.4:
                    print(f"🚫 Skipping {os.path.basename(pdf_path)} "
                          f"→ valid_ratio={valid_ratio:.2f}")
                    return []

        except Exception as e:
            print(f"⚠️ Error processing {pdf_path}: {e}")

        return pages

    # ==========================================================
    # PDF → PDF Search (page-based semantic comparison)
    # ==========================================================
    def search_similar_pdfs(self, query_pdf, folder="./data/pdfs", top_k=5):
        query_pages = self.get_pdf_pages_embeddings(query_pdf)
        if not query_pages:
            print("❌ No valid text found in the query PDF.")
            return []

        results = []
        pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]

        for file in tqdm(pdf_files, desc="Comparing PDFs"):
            pdf_path = os.path.join(folder, file)
            if os.path.abspath(pdf_path) == os.path.abspath(query_pdf):
                continue

            target_pages = self.get_pdf_pages_embeddings(pdf_path)
            if not target_pages:
                continue

            best_score, best_page, best_snippet = 0, None, ""
            for q_page_num, q_emb, q_text in query_pages:
                for t_page_num, t_emb, t_text in target_pages:
                    score = util.cos_sim(q_emb, t_emb).item()
                    overlap = word_overlap(q_text, t_text)

                    # Require at least 5 shared words to consider as relevant
                    if score > best_score and overlap >= 5:
                        best_score = score
                        best_page = t_page_num
                        best_snippet = t_text[:300].replace("\n", " ") + "..."

            if best_score >= 0.96:
                results.append({
                    "file": file,
                    "score": best_score,
                    "page": best_page,
                    "snippet": best_snippet
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ==========================================================
    # TEXT → PDF Search
    # ==========================================================
    def search_by_text(self, query_text, folder="./data/pdfs", top_k=5):
        print(f"\n💬 Text query: {query_text}")
        query_emb = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        results = []

        pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]

        for file in tqdm(pdf_files, desc="Searching PDFs"):
            pdf_path = os.path.join(folder, file)
            best_score = 0
            best_page = None
            best_snippet = ""

            try:
                with fitz.open(pdf_path) as doc:
                    for i, page in enumerate(doc):
                        page_text = page.get_text("text").strip()
                        if len(page_text) < self.min_chars:
                            continue

                        emb = self.model.encode(page_text, convert_to_tensor=True, normalize_embeddings=True)
                        score = util.cos_sim(query_emb, emb).item()

                        if score > best_score:
                            best_score = score
                            best_page = i + 1
                            best_snippet = page_text[:300].replace("\n", " ") + "..."

            except Exception as e:
                print(f"⚠️ Error reading {pdf_path}: {e}")
                continue

            if best_score >= self.min_score:
                results.append({
                    "file": file,
                    "score": best_score,
                    "page": best_page,
                    "snippet": best_snippet
                })

        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            print("❌ No relevant matches found.")
        else:
            print(f"✅ Found {len(results)} relevant PDFs (≥ {self.min_score:.2f})")

        return results[:top_k]

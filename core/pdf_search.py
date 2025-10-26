import os
import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
    return text.strip()


class PDFSearcher:
    """
    PDF Similarity Searcher using a fine-tuned M-CLIP model.
    Supports semantic text-based comparison between PDFs.
    """

    def __init__(self, model_path="./models/mclip_finetuned_coco_ready", device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Loading M-CLIP model from: {model_path} on {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)

    # -------------------------------------
    # Compute embedding of PDF text
    # -------------------------------------
    def get_pdf_embedding(self, pdf_path):
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"⚠️ No text found in {pdf_path}")
            return None
        emb = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return emb

    # -------------------------------------
    # Compare PDFs by semantic similarity
    # -------------------------------------
    def search_similar_pdfs(self, query_pdf, folder="./data/pdfs", top_k=5):
        print(f"📘 Query PDF: {query_pdf}")
        query_emb = self.get_pdf_embedding(query_pdf)
        if query_emb is None:
            print("❌ No text found in query PDF.")
            return []

        results = []
        for file in tqdm(os.listdir(folder), desc="Comparing PDFs"):
            if not file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(folder, file)
            if os.path.abspath(pdf_path) == os.path.abspath(query_pdf):
                continue

            emb = self.get_pdf_embedding(pdf_path)
            if emb is None:
                continue

            score = util.cos_sim(query_emb, emb).item()
            results.append((file, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # -------------------------------------
    # Compare PDFs by text
    # -------------------------------------
    def search_by_text(self, query_text, folder="./data/pdfs", top_k=5):
        print(f"💬 Text query: {query_text}")
        query_emb = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)

        results = []

        for file in tqdm(os.listdir(folder), desc="Searching PDFs"):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(folder, file)
            best_score = -1
            best_snippet = ""
            best_page = None

            try:
                with fitz.open(pdf_path) as doc:
                    for i, page in enumerate(doc):
                        page_text = page.get_text("text").strip()
                        if not page_text:
                            continue

                        emb = self.model.encode(page_text, convert_to_tensor=True, normalize_embeddings=True)
                        score = util.cos_sim(query_emb, emb).item()

                        if score > best_score:
                            best_score = score
                            best_snippet = page_text[:300].replace("\n", " ") + "..."
                            best_page = i + 1  # page numbering starts at 1
            except Exception as e:
                print(f"⚠️ Error reading {pdf_path}: {e}")
                continue

            if best_score > 0:
                results.append({
                    "file": file,
                    "score": best_score,
                    "page": best_page,
                    "snippet": best_snippet
                })

        # Ταξινόμηση με βάση το score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Επιστροφή top_k αποτελεσμάτων
        return results[:top_k]


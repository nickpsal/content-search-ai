import sqlite3
import numpy as np
import torch
import re
import fitz
from sentence_transformers import SentenceTransformer


class PDFSearcher:
    """
    PDF Searcher — SQLite Powered
    """

    def __init__(self, db_path="content_search_ai.db", model_path="./models/mclip_finetuned_coco_ready"):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🧠 Loading M-CLIP model from {model_path} on {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)

        self.min_score = 0.90
        self.min_chars = 50   # 🔥 ΑΠΑΡΑΙΤΗΤΟ

    # =======================================
    # LOAD ALL EMBEDDINGS FROM SQLITE
    # =======================================
    def _load_pdf_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT pdf_path, page_number, text_content, embedding FROM pdf_pages")
        rows = cursor.fetchall()
        conn.close()

        pdf_pages = []
        for pdf_path, page_number, text_content, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            pdf_pages.append({
                "pdf_path": pdf_path,
                "page": page_number,
                "text": text_content,
                "vector": vec
            })

        return pdf_pages

    # =======================================
    # EXTRACT EMBEDDINGS FOR PDF → DB
    # =======================================
    def get_pdf_pages_embeddings(self, pdf_path):
        pages = []
        valid_pages = 0

        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)

                for i, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if not text or len(text) < self.min_chars:
                        continue

                    clean_text = re.sub(r"[^A-Za-zΑ-Ωα-ω\s]", " ", text)
                    words = [w for w in clean_text.split() if len(w) > 2]
                    if len(words) < 10:
                        continue

                    letters = sum(c.isalpha() for c in text)
                    ratio = letters / (len(text) + 1)
                    if ratio < 0.6:
                        continue

                    valid_pages += 1

                    emb = self.model.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    ).astype(np.float32)

                    pages.append((i + 1, emb, text))

                if valid_pages < 1:
                    print("🚫 Low text density, skipping.")
                    return []

        except Exception as e:
            print(f"⚠️ Error processing {pdf_path}: {e}")

        return pages

    # =======================================
    # TEXT → PDF SEARCH
    # =======================================
    def search_by_text(self, query_text, top_k=5):
        pdf_pages = self._load_pdf_embeddings()
        if not pdf_pages:
            print("❌ No PDF embeddings found.")
            return []

        query_vec = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        results = []
        for page in pdf_pages:
            v = page["vector"]

            sim = np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v))

            if sim >= self.min_score:
                results.append({
                    "pdf": page["pdf_path"],
                    "score": float(sim),
                    "page": page["page"],
                    "snippet": page["text"][:300].replace("\n", " ") + "..."
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

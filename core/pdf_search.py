import sqlite3
import numpy as np
import torch
import re
import fitz
from sentence_transformers import SentenceTransformer

def split_into_paragraphs(text, min_len=80):
    raw = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw if len(p.strip()) >= min_len]
    return paragraphs

class PDFSearcher:
    """
    PDF Searcher — Pure Semantic Retrieval
    Page-level embeddings with adaptive thresholding.
    """

    def __init__(
        self,
        db_path="content_search_ai.db",
        model_path="./models/mclip_finetuned_coco_ready"
    ):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading M-CLIP model from {model_path} on {self.device}")

        self.model = SentenceTransformer(
            model_path,
            device=self.device,
            model_kwargs={
                "device_map": None,
                "low_cpu_mem_usage": False,
                "torch_dtype": torch.float32
            }
        )

        # used only during indexing
        self.min_chars = 50

    # ==================================================
    # LOAD PDF PAGE EMBEDDINGS FROM SQLITE
    # ==================================================
    def _load_pdf_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT pdf_path, page_number, text_content, embedding FROM pdf_pages"
        )
        rows = cursor.fetchall()
        conn.close()

        pages = []
        for pdf_path, page_number, text_content, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            pages.append({
                "pdf_path": pdf_path,
                "page": page_number,
                "text": text_content,
                "vector": vec
            })

        return pages

    # ==================================================
    # EXTRACT EMBEDDINGS FOR PDF (INDEXING ONLY)
    # ==================================================
    def get_pdf_pages_embeddings(self, pdf_path):
        pages = []
        valid_pages = 0

        try:
            with fitz.open(pdf_path) as doc:
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
                    print("[INFO] Low text density, skipping PDF.")
                    return []

        except Exception as e:
            print(f"[WARN] Error processing {pdf_path}: {e}")

        return pages

    # ==================================================
    # TEXT → PDF SEARCH (PURE RETRIEVAL)
    # ==================================================
    def search_by_text(self, query_text, top_k=5):
        pdf_pages = self._load_pdf_embeddings()
        if not pdf_pages:
            return []

        query_vec = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # --- similarity distribution ---
        sims = np.array([
            float(np.dot(query_vec, page["vector"]))
            for page in pdf_pages
        ])

        mean, std = sims.mean(), sims.std()
        MIN_SIM = mean + 0.3 * std  # ✅ adaptive

        results = []

        for page, sim in zip(pdf_pages, sims):
            if sim < MIN_SIM:
                continue

            # -------- paragraph explainability --------
            paragraphs = split_into_paragraphs(page["text"])
            best_para = None
            best_para_score = None

            if paragraphs:
                para_embs = self.model.encode(
                    paragraphs,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                para_sims = para_embs @ query_vec
                idx = int(np.argmax(para_sims))

                if para_sims[idx] >= MIN_SIM:
                    best_para = paragraphs[idx]
                    best_para_score = float(para_sims[idx])

            # ------------------------------------------

            confidence = (sim - MIN_SIM) / (1 - MIN_SIM)
            confidence = float(np.clip(confidence, 0.0, 1.0))

            results.append({
                "pdf": page["pdf_path"],
                "page": page["page"],
                "score": sim,
                "confidence": confidence,
                "matched_paragraph": best_para,
                "paragraph_score": best_para_score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]



import os
import sqlite3
import numpy as np
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import time


# =========================================================
# QUERY NORMALIZATION (μόνο EL → EN, ΟΧΙ semantics)
# =========================================================
def normalize_text_image_query(query: str, lang: str) -> str:
    query = query.strip().lower()

    if lang == "el":
        mappings = {
            "άντρας": "man",
            "γυναίκα": "woman",
            "μηχανή": "motorcycle",
            "μοτοσυκλέτα": "motorcycle",
            "καβαλάει": "riding",
            "οδηγεί": "riding",
            "πάνω σε": "on",
        }
        for gr, en in mappings.items():
            query = query.replace(gr, en)

    return query


def build_prompt_variants(query: str):
    return list(dict.fromkeys([
        f"a photo of {query}",
        f"a picture of {query}",
        f"a photo showing {query}",
        query
    ]))


def detect_language(text: str):
    for ch in text:
        if 'α' <= ch <= 'ω' or 'Α' <= ch <= 'Ω':
            return "el"
    return "en"


# =========================================================
# IMAGE SEARCHER (GENERIC)
# =========================================================
class ImageSearcher:

    def __init__(self, db_path="content_search_ai.db"):
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_model = SentenceTransformer(
            "./models/mclip_finetuned_coco_ready",
            device=self.device
        )

    # -----------------------------------------------------
    def _load_sqlite_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, filepath, embedding FROM images")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "filename": filename,
                "path": image_path,
                "vector": np.frombuffer(blob, dtype=np.float32)
            }
            for filename, image_path, blob in rows
        ]

    # -----------------------------------------------------
    # TEXT → IMAGE (GENERIC RETRIEVAL)
    # -----------------------------------------------------
    def search(self, query: str, top_k=5):
        images = self._load_sqlite_embeddings()
        if not images:
            raise RuntimeError("No image embeddings found")

        lang = detect_language(query)
        query = normalize_text_image_query(query, lang)
        prompts = build_prompt_variants(query)

        query_embs = self.text_model.encode(
            prompts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # -----------------------------------
        # 1️⃣ Coarse scoring
        # -----------------------------------
        scores = []
        for img in images:
            sim = max(float(np.dot(q, img["vector"])) for q in query_embs)
            scores.append(sim)

        scores = np.array(scores)
        mean, std = scores.mean(), scores.std()

        # adaptive threshold (GENERIC)
        MIN_SIM = mean + 0.3 * std

        # -----------------------------------
        # 2️⃣ Final ranking
        # -----------------------------------
        results = []
        for img, sim in zip(images, scores):
            if sim < MIN_SIM:
                continue

            confidence = (sim - mean) / (std + 1e-6)
            confidence = max(0.0, min(confidence, 1.0))

            results.append({
                "filename": img["filename"],
                "score": float(sim),
                "confidence": float(confidence),
                "path": img["path"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # -----------------------------------------------------
    # IMAGE → IMAGE (unchanged, generic)
    # -----------------------------------------------------
    def search_by_image(self, query_image_path: str, top_k=5):
        images = self._load_sqlite_embeddings()

        image = self.preprocess(
            Image.open(query_image_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.image_model.encode_image(image)
            q /= q.norm(dim=-1, keepdim=True)
            q = q.cpu().numpy().flatten()

        results = []
        for img in images:
            sim = float(np.dot(q, img["vector"]))
            results.append({
                "filename": img["filename"],
                "score": sim,
                "path": img["path"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

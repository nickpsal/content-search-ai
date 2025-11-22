import os
import sqlite3
import numpy as np
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import time


class ImageSearcher:
    """
    SQLite-powered Image Searcher
    Supports:
      • text → image
      • image → image
    Uses:
      • M-CLIP for text
      • CLIP ViT-B/32 for images
      • SQLite for embedding storage
    """

    def __init__(self, db_path="content_search_ai.db"):
        self.db_path = db_path

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP image encoder
        self.image_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Load M-CLIP text encoder
        self.text_model = SentenceTransformer("./models/mclip_finetuned_coco_ready",
                                              device=self.device)

    # -------------------------------------------------------------
    # INTERNAL: Load all image embeddings from SQLite
    # -------------------------------------------------------------
    def _load_sqlite_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT filename, image_path, embedding FROM images")
        rows = cursor.fetchall()
        conn.close()

        embeddings = {}

        for filename, image_path, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            embeddings[filename] = {
                "path": image_path,
                "vector": vec
            }

        return embeddings

    # -------------------------------------------------------------
    # TEXT → IMAGE
    # -------------------------------------------------------------
    def search(self, query: str, top_k=5):
        embeddings = self._load_sqlite_embeddings()

        if not embeddings:
            raise Exception("❌ No embeddings found in SQLite!")

        # Encode text query with M-CLIP
        query_emb = self.text_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        results = []
        start = time.time()

        for filename, item in embeddings.items():
            img_vec = item["vector"]

            # Cosine similarity
            sim = np.dot(query_emb, img_vec) / (
                np.linalg.norm(query_emb) * np.linalg.norm(img_vec)
            )

            results.append({
                "filename": filename,
                "score": float(sim),
                "path": item["path"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        print(f"⏱️ Text → Image search completed in {time.time() - start:.2f}s")
        return results[:top_k]

    # -------------------------------------------------------------
    # IMAGE → IMAGE
    # -------------------------------------------------------------
    def search_by_image(self, query_image_path: str, top_k=5):
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"❌ File not found: {query_image_path}")

        embeddings = self._load_sqlite_embeddings()

        # Encode query image with CLIP
        image = self.preprocess(Image.open(query_image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.image_model.encode_image(image)
            query_emb /= query_emb.norm(dim=-1, keepdim=True)
            query_emb = query_emb.cpu().numpy().flatten().astype(np.float32)

        results = []
        start = time.time()

        for filename, item in embeddings.items():
            img_vec = item["vector"]

            # Cosine similarity
            sim = np.dot(query_emb, img_vec) / (
                np.linalg.norm(query_emb) * np.linalg.norm(img_vec)
            )

            results.append({
                "filename": filename,
                "score": float(sim),
                "path": item["path"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        print(f"⏱️ Image → Image search completed in {time.time() - start:.2f}s")
        return results[:top_k]

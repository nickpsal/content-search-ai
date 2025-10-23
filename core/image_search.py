import os
import json
import torch
import clip
import requests
import zipfile
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True  # αποφυγή σφαλμάτων σε κατεστραμμένες εικόνες


def download_and_extract(url, dest_zip, extract_to):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_zip, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_zip)}",
        total=total, unit='B', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


class ImageSearcher:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images", "val2017")
        self.extra_image_dir = os.path.join(data_dir, "images", "other")
        self.caption_file = os.path.join(data_dir, "annotations", "annotations", "captions_val2017.json")

        self.image_embed_path = os.path.join(data_dir, "embeddings", "val2017_image_embeddings.pt")
        self.extra_image_embed_path = os.path.join(data_dir, "embeddings", "other_image_embeddings.pt")
        self.text_embed_path = os.path.join(data_dir, "embeddings", "coco_val_text_embeddings.pt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "./models//mclip_finetuned_coco_ready"
        # self.model = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

        # HYBRID MODE
        print("🚀 Using hybrid setup:")
        print(f"   🧠 Text encoder: {self.model}")
        print("   🖼️ Image encoder: OpenAI CLIP ViT-B/32")

        self.text_model = SentenceTransformer(self.model, device=self.device)
        self.image_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    # ---------------------------------------------------------------------
    # DOWNLOAD COCO DATA
    # ---------------------------------------------------------------------
    def download_coco_data(self):
        annotations_dir = os.path.join(self.data_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        caption_zip = os.path.join(annotations_dir, "annotations_trainval2017.zip")
        if not os.path.exists(self.caption_file):
            print("📦 Downloading captions...")
            download_and_extract(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                caption_zip,
                annotations_dir
            )

        images_dir = os.path.join(self.data_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        image_zip = os.path.join(images_dir, "val2017.zip")
        if not os.path.exists(self.image_dir) or len(os.listdir(self.image_dir)) == 0:
            print("🖼️ Downloading images...")
            download_and_extract(
                "http://images.cocodataset.org/zips/val2017.zip",
                image_zip,
                images_dir
            )
        else:
            print(f"✅ Images already exist in {self.image_dir}")

    # ---------------------------------------------------------------------
    # EXTRACT IMAGE EMBEDDINGS (HYBRID: OpenAI CLIP)
    # ---------------------------------------------------------------------
    def extract_image_embeddings(self, folder="val2017", force=False):
        if folder == "val2017":
            img_dir = self.image_dir
            embed_path = self.image_embed_path
        else:
            img_dir = self.extra_image_dir
            embed_path = self.extra_image_embed_path

        if os.path.exists(embed_path) and not force:
            print(f"✅ Embeddings already exist at {embed_path}")
            return

        if not os.path.exists(img_dir):
            print(f"❌ Folder not found: {img_dir}")
            return

        image_paths = [
            os.path.join(img_dir, name)
            for name in os.listdir(img_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"📸 Found {len(image_paths)} images in {img_dir}")

        embeddings = {}
        for path in tqdm(image_paths, desc=f"Extracting embeddings from {folder}"):
            try:
                image = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.image_model.encode_image(image)
                    features = features / features.norm(dim=-1, keepdim=True)
                embeddings[os.path.basename(path)] = features.cpu()
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")
                continue

        if not embeddings:
            print("❌ No embeddings were created.")
            return

        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        torch.save(embeddings, embed_path)
        print(f"✅ Saved {len(embeddings)} image embeddings to {embed_path}")

    # ---------------------------------------------------------------------
    # EXTRACT TEXT EMBEDDINGS (HYBRID: multilingual encoder)
    # ---------------------------------------------------------------------
    def extract_text_embeddings(self, force=False):
        if os.path.exists(self.text_embed_path) and not force:
            print(f"✅ Caption embeddings already exist at {self.text_embed_path}")
            return

        with open(self.caption_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = {}
        for ann in tqdm(data["annotations"], desc="Extracting caption embeddings"):
            caption = ann["caption"]
            image_name = f"{ann['image_id']:012}.jpg"
            features = self.text_model.encode(caption, convert_to_tensor=True, normalize_embeddings=True)
            embeddings.setdefault(image_name, []).append(features.cpu())

        final_embeddings = {k: torch.stack(v).mean(dim=0) for k, v in embeddings.items()}
        os.makedirs(os.path.dirname(self.text_embed_path), exist_ok=True)
        torch.save(final_embeddings, self.text_embed_path)
        print(f"✅ Saved {len(final_embeddings)} caption embeddings to {self.text_embed_path}")

    # ---------------------------------------------------------------------
    # SEARCH TEXT → IMAGE
    # ---------------------------------------------------------------------
    def search(self, query: str, top_k=5, log_file="search_log.txt", verbose=True):
        # φόρτωση embeddings (COCO + OTHER)
        image_embeddings = {}
        if os.path.exists(self.image_embed_path):
            image_embeddings.update(torch.load(self.image_embed_path, weights_only=True))
        if os.path.exists(self.extra_image_embed_path):
            image_embeddings.update(torch.load(self.extra_image_embed_path, weights_only=True))

        if not image_embeddings:
            raise FileNotFoundError("❌ No image embeddings found. Run extract_image_embeddings() first.")

        print(f"✅ Loaded {len(image_embeddings)} image embeddings")

        # text encoding
        text_features = self.text_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

        results = []
        start_time = time.time()

        # verbose logging for calculation
        if verbose:
            print("\n📐 Cosine Similarity Formula: sim(A,B) = (A · B) / (||A|| * ||B||)")
            print("Since normalize_embeddings=True ⇒ ||A|| = ||B|| = 1 ⇒ sim(A,B) = A · B\n")

        for name, img_emb in image_embeddings.items():
            similarity = torch.cosine_similarity(text_features, img_emb, dim=-1)

            if verbose:
                print(f"Name => Similarity: {name} => {similarity.item():.4f}")

            path = os.path.join(self.image_dir, name)
            if not os.path.exists(path):
                path = os.path.join(self.extra_image_dir, name)

            results.append({
                "path": path,
                "score": similarity.item(),
                "name": name
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        elapsed = time.time() - start_time

        # εμφάνιση top-k
        print(f"\n🔎 Top {top_k} results for: {query}\n")
        for i, r in enumerate(results[:top_k]):
            print(f"{i + 1}. {r['path']}  (score={r['score']:.4f})")
        print(f"\n⏱️ Search completed in {elapsed:.2f}s")

        # ✍️ Καταγραφή σε log file (append mode)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"🕓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Total Images: {len(image_embeddings)}\n")
            f.write(f"Search Time: {elapsed:.2f}s\n")
            f.write("Similarity Calculation:\n")
            f.write("  sim(A,B) = (A · B) / (||A|| * ||B||)\n")
            f.write("  Since normalize_embeddings=True, ||A||=||B||=1 ⇒ sim(A,B) = A · B\n\n")
            f.write(f"Top {top_k} Results:\n")
            for i, r in enumerate(results[:top_k]):
                f.write(f"  {i + 1}. {r['name']} (score={r['score']:.4f})\n")
            f.write("\nAll Similarities (Top 20 by score):\n")
            for r in results[:20]:
                f.write(f"  {r['name']} => {r['score']:.4f}\n")
            f.write("\n\n")

        return results[:top_k]

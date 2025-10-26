import os
import json
import torch
import clip
import requests
import zipfile
from PIL import Image, ImageFile
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datetime import datetime
import time

# Allow loading of truncated or corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def download_and_extract(url, dest_zip, extract_to):
    """Download and extract a ZIP file with progress bar."""
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
    """
    Image Similarity Searcher using hybrid CLIP (OpenAI) + M-CLIP encoders.
    Supports text-to-image and image-to-image retrieval.
    """

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images", "val2017")
        self.extra_image_dir = os.path.join(data_dir, "images", "other")
        self.caption_file = os.path.join(data_dir, "annotations", "annotations", "captions_val2017.json")

        self.image_embed_path = os.path.join(data_dir, "embeddings", "val2017_image_embeddings.pt")
        self.extra_image_embed_path = os.path.join(data_dir, "embeddings", "other_image_embeddings.pt")
        self.text_embed_path = os.path.join(data_dir, "embeddings", "coco_val_text_embeddings.pt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "./models/mclip_finetuned_coco_ready"

        # Hybrid model setup
        self.text_model = SentenceTransformer(self.model, device=self.device)
        self.image_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    # ==========================================================
    # Download COCO Dataset
    # ==========================================================
    def download_coco_data(self):
        annotations_dir = os.path.join(self.data_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        caption_zip = os.path.join(annotations_dir, "annotations_trainval2017.zip")

        if not os.path.exists(self.caption_file):
            print("📦 Downloading COCO captions...")
            download_and_extract(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                caption_zip,
                annotations_dir
            )

        images_dir = os.path.join(self.data_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        image_zip = os.path.join(images_dir, "val2017.zip")

        if not os.path.exists(self.image_dir) or not os.listdir(self.image_dir):
            print("🖼️ Downloading COCO validation images...")
            download_and_extract(
                "http://images.cocodataset.org/zips/val2017.zip",
                image_zip,
                images_dir
            )
        else:
            print(f"✅ Images already available at {self.image_dir}")

    # ==========================================================
    # Extract Image Embeddings (CLIP Encoder)
    # ==========================================================
    def extract_image_embeddings(self, folder="val2017", force=False):
        if folder == "val2017":
            img_dir = self.image_dir
            embed_path = self.image_embed_path
        else:
            img_dir = self.extra_image_dir
            embed_path = self.extra_image_embed_path

        if os.path.exists(embed_path) and not force:
            print(f"✅ Image embeddings already exist at {embed_path}")
            return

        if not os.path.exists(img_dir):
            print(f"❌ Image folder not found: {img_dir}")
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
                    features /= features.norm(dim=-1, keepdim=True)
                embeddings[os.path.basename(path)] = features.cpu()
            except Exception as e:
                print(f"⚠️ Skipped {os.path.basename(path)} due to error: {e}")

        if not embeddings:
            print("❌ No embeddings were generated.")
            return

        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        torch.save(embeddings, embed_path)
        print(f"✅ Saved {len(embeddings)} embeddings → {embed_path}")

    # ==========================================================
    # Extract Text Embeddings (M-CLIP Encoder)
    # ==========================================================
    def extract_text_embeddings(self, force=False):
        if os.path.exists(self.text_embed_path) and not force:
            print(f"✅ Text embeddings already exist at {self.text_embed_path}")
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
        print(f"✅ Saved {len(final_embeddings)} caption embeddings → {self.text_embed_path}")

    # ==========================================================
    # Text → Image Search
    # ==========================================================
    def search(self, query: str, top_k=5, verbose=False):
        image_embeddings = {}
        if os.path.exists(self.image_embed_path):
            image_embeddings.update(torch.load(self.image_embed_path, weights_only=True))
        if os.path.exists(self.extra_image_embed_path):
            image_embeddings.update(torch.load(self.extra_image_embed_path, weights_only=True))

        if not image_embeddings:
            raise FileNotFoundError("❌ No image embeddings found. Run extract_image_embeddings() first.")

        print(f"✅ Loaded {len(image_embeddings)} embeddings")
        query_emb = self.text_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

        results = []
        start_time = time.time()
        for name, img_emb in image_embeddings.items():
            similarity = torch.cosine_similarity(query_emb, img_emb, dim=-1)
            path = os.path.join(self.image_dir, name)
            if not os.path.exists(path):
                path = os.path.join(self.extra_image_dir, name)
            results.append({"path": path, "score": similarity.item(), "name": name})

        results.sort(key=lambda x: x["score"], reverse=True)
        elapsed = time.time() - start_time
        print(f"⏱️ Search completed in {elapsed:.2f}s")
        return results[:top_k]

    # ==========================================================
    # Image → Image Search
    # ==========================================================
    def search_by_image(self, query_image_path: str, top_k=5):
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"❌ Query image not found: {query_image_path}")

        image_embeddings = {}
        if os.path.exists(self.image_embed_path):
            image_embeddings.update(torch.load(self.image_embed_path, weights_only=True))
        if os.path.exists(self.extra_image_embed_path):
            image_embeddings.update(torch.load(self.extra_image_embed_path, weights_only=True))

        if not image_embeddings:
            raise FileNotFoundError("❌ No image embeddings found. Run extract_image_embeddings() first.")

        print(f"✅ Loaded {len(image_embeddings)} embeddings")

        # Encode the query image
        image = self.preprocess(Image.open(query_image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.image_model.encode_image(image)
            query_emb /= query_emb.norm(dim=-1, keepdim=True)

        # Compute similarities
        results = []
        start_time = time.time()
        for name, img_emb in image_embeddings.items():
            similarity = torch.cosine_similarity(query_emb, img_emb, dim=-1)
            path = os.path.join(self.image_dir, name)
            if not os.path.exists(path):
                path = os.path.join(self.extra_image_dir, name)
            results.append({"path": path, "score": similarity.item(), "name": name})

        results.sort(key=lambda x: x["score"], reverse=True)
        elapsed = time.time() - start_time
        print(f"⏱️ Image-to-Image search completed in {elapsed:.2f}s")

        return results[:top_k]

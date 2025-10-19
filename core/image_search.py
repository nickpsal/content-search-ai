import os
import json
import torch
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer


def download_and_extract(url, dest_zip, extract_to):
    """Κατεβάζει και αποσυμπιέζει αρχεία με progress bar"""
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


def translate_query(query: str, target_lang="en"):
    """Μεταφράζει ερώτημα εάν δεν είναι στα αγγλικά"""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(query)
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return query


class ImageSearcher:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images", "val2017")
        self.caption_file = os.path.join(data_dir, "annotations", "annotations", "captions_val2017.json")
        self.image_embed_path = os.path.join(data_dir, "embeddings", "coco_val_image_embeddings.pt")
        self.text_embed_path = os.path.join(data_dir, "embeddings", "coco_val_text_embeddings.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ M-CLIP μοντέλο (multilingual CLIP)
        self.model = SentenceTransformer("./models/mclip_finetuned_coco", device=self.device)

    # ---------------------------------------------------------
    # 🧠 ENCODERS
    # ---------------------------------------------------------
    def encode_text(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, device=self.device)

    def encode_image(self, image):
        return self.model.encode(image, convert_to_tensor=True, device=self.device)

    # ---------------------------------------------------------
    # 📦 DOWNLOAD COCO DATA
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 🧩 EXTRACT IMAGE EMBEDDINGS
    # ---------------------------------------------------------
    def extract_image_embeddings(self, force=False):
        if os.path.exists(self.image_embed_path) and not force:
            print(f"✅ Image embeddings already exist at {self.image_embed_path}")
            return

        embeddings = {}
        image_paths = [os.path.join(self.image_dir, name)
                       for name in os.listdir(self.image_dir) if name.endswith(".jpg")]

        print(f"📸 Found {len(image_paths)} images.")
        for path in tqdm(image_paths, desc="Extracting image embeddings"):
            try:
                image = Image.open(path).convert("RGB")
                features = self.model.encode(image, convert_to_tensor=True, device=self.device)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings[os.path.basename(path)] = features.cpu()
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

        os.makedirs(os.path.dirname(self.image_embed_path), exist_ok=True)
        torch.save(embeddings, self.image_embed_path)
        print(f"✅ Saved {len(embeddings)} image embeddings to {self.image_embed_path}")

    # ---------------------------------------------------------
    # ✍️ EXTRACT TEXT EMBEDDINGS
    # ---------------------------------------------------------
    def extract_text_embeddings(self, force=False):
        if os.path.exists(self.text_embed_path) and not force:
            print(f"✅ Caption embeddings already exist at {self.text_embed_path}")
            return

        with open(self.caption_file, 'r') as f:
            data = json.load(f)

        embeddings = {}
        for ann in tqdm(data['annotations'], desc="Extracting caption embeddings"):
            caption = ann['caption']
            image_name = f"{ann['image_id']:012}.jpg"

            features = self.model.encode(caption, convert_to_tensor=True, device=self.device)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.setdefault(image_name, []).append(features.cpu())

        final_embeddings = {k: torch.stack(v).mean(dim=0) for k, v in embeddings.items()}
        os.makedirs(os.path.dirname(self.text_embed_path), exist_ok=True)
        torch.save(final_embeddings, self.text_embed_path)
        print(f"✅ Saved {len(final_embeddings)} caption embeddings to {self.text_embed_path}")

    # ---------------------------------------------------------
    # 🔍 SEARCH FUNCTION
    # ---------------------------------------------------------
    def search(self, query: str, top_k=5):
        if not os.path.exists(self.image_embed_path):
            raise FileNotFoundError("❌ Image embeddings not found. Run extract_image_embeddings() first.")

        image_embeddings = torch.load(self.image_embed_path, weights_only=True)

        # Μετάφραση (αν χρειάζεται)
        translated_query = translate_query(query)
        print(f"🔎 Searching for: \"{translated_query}\"")

        # Text embedding (πολυγλωσσικό)
        with torch.no_grad():
            text_features = self.model.encode(translated_query, convert_to_tensor=True, device=self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        results = []
        for name, img_emb in image_embeddings.items():
            similarity = torch.cosine_similarity(text_features, img_emb, dim=-1)
            results.append((name, similarity.item()))

        results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🏆 Top {top_k} results:")
        top_results = []
        for i in range(top_k):
            name, score = results[i]
            print(f"{i + 1}. {name} (score: {score:.4f})")
            img_path = os.path.join(self.image_dir, name)
            top_results.append({"path": img_path, "score": round(score, 4)})

        return top_results

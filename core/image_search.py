import os
import json
import torch
import clip
import requests
import zipfile
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def download_and_extract(url, dest_zip, extract_to):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_zip, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(dest_zip)}",
            total=total, unit='B', unit_scale=True, unit_divisor=1024) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

class ImageSearcher:
    def __init__(self, data_dir="./data", model_name="./models/mclip_finetuned_coco", use_finetuned=True):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images", "val2017")
        self.caption_file = os.path.join(data_dir, "annotations", "annotations", "captions_val2017.json")
        self.image_embed_path = os.path.join(data_dir, "embeddings", "coco_val_image_embeddings.pt")
        self.text_embed_path = os.path.join(data_dir, "embeddings", "coco_val_text_embeddings.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_finetuned = use_finetuned

        # ✅ εδώ γίνεται η μόνη αλλαγή:
        if self.use_finetuned:
            print(f"🚀 Using fine-tuned M-CLIP model from {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.preprocess = None
        else:
            print(f"🚀 Using OpenAI CLIP model: {model_name}")
            self.model, self.preprocess = clip.load(model_name, device=self.device)

    def download_coco_data(self):
        # (καμία αλλαγή)
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
            print(f"Images Already exists on {images_dir} folder")

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
                if self.use_finetuned:
                    image = Image.open(path).convert("RGB")
                    features = self.model.encode(image, convert_to_tensor=True, normalize_embeddings=True)
                else:
                    image = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        features = self.model.encode_image(image)
                        features = features / features.norm(dim=-1, keepdim=True)
                embeddings[os.path.basename(path)] = features.cpu()
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

        os.makedirs(os.path.dirname(self.image_embed_path), exist_ok=True)
        torch.save(embeddings, self.image_embed_path)
        print(f"✅ Saved {len(embeddings)} image embeddings to {self.image_embed_path}")

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

            if self.use_finetuned:
                features = self.model.encode(caption, convert_to_tensor=True, normalize_embeddings=True)
            else:
                text = clip.tokenize(caption).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_text(text)
                    features = features / features.norm(dim=-1, keepdim=True)

            embeddings.setdefault(image_name, []).append(features.cpu())

        final_embeddings = {
            k: torch.stack(v).mean(dim=0) for k, v in embeddings.items()
        }

        os.makedirs(os.path.dirname(self.text_embed_path), exist_ok=True)
        torch.save(final_embeddings, self.text_embed_path)
        print(f"✅ Saved {len(final_embeddings)} caption embeddings to {self.text_embed_path}")

    def search(self, query: str, top_k=5):
        if not os.path.exists(self.image_embed_path):
            raise FileNotFoundError("❌ Image embeddings not found.")

        image_embeddings = torch.load(self.image_embed_path, weights_only=True)

        if self.use_finetuned:
            text_features = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        else:
            text = clip.tokenize([query]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        results = []
        for name, img_emb in image_embeddings.items():
            similarity = torch.cosine_similarity(text_features, img_emb, dim=-1)
            results.append({
                "path": os.path.join(self.image_dir, name),
                "score": similarity.item(),
                "name": name
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        print(f"{len(results)} Images founded")

        return results[:min(top_k, len(results))]

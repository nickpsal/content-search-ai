import json
import os
import sys
import zipfile
import clip
import requests
import torch
from PIL import Image
from deep_translator import GoogleTranslator
from tqdm import tqdm

# ---------------------------------------- Download and Extract Data ---------------------------------------- #
def download_and_extract(url, dest_zip, extract_to):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_zip, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_zip)}",
        total=total, unit='B', unit_scale=True, unit_divisor=1024,
        file=sys.stdout
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# ---------------------------------------- Translate Query to English ---------------------------------------- #
def translate_query(query: str, target_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(query)
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return query  # fallback

# ---------------------------------------------- Image Searcher ---------------------------------------------- #
class ImageSearcher:
    def __init__(self, data_dir, model_name="ViT-B/32"):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images", "val2017")
        self.caption_file = os.path.join(data_dir, "annotations", "annotations", "captions_val2017.json")
        self.image_embed_path = os.path.join(data_dir, "embeddings", "coco_val_image_embeddings.pt")
        self.text_embed_path = os.path.join(data_dir, "embeddings", "coco_val_text_embeddings.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

# -------------------------------------------- Download Coco Data -------------------------------------------- #
    def download_coco_data(self):
        # Captions
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

        # Images
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
            print(f"✅ Images already exist in {images_dir}")

# ---------------------------------------- Download Image Embeddings ---------------------------------------- #
    def extract_image_embeddings(self, force=False):
        if os.path.exists(self.image_embed_path) and not force:
            print(f"✅ Image embeddings already exist at {self.image_embed_path}")
            return

        embeddings = {}
        image_paths = [os.path.join(self.image_dir, name)
                       for name in os.listdir(self.image_dir) if name.endswith(".jpg")]

        print(f"📸 Found {len(image_paths)} images.")

        for path in tqdm(image_paths, desc="Extracting image embeddings", file=sys.stdout):
            try:
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

# ---------------------------------------- Download Text Embeddings ---------------------------------------- #
    def extract_text_embeddings(self, force=False):
        if os.path.exists(self.text_embed_path) and not force:
            print(f"✅ Caption embeddings already exist at {self.text_embed_path}")
            return

        with open(self.caption_file, 'r') as f:
            data = json.load(f)

        embeddings = {}

        for ann in tqdm(data['annotations'], desc="Extracting caption embeddings", file=sys.stdout):
            caption = ann['caption']
            image_name = f"{ann['image_id']:012}.jpg"

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

# ----------------------------------------- Search Image by Query ----------------------------------------- #
    def search_by_query(self, query: str, top_k=6):
        if not os.path.exists(self.image_embed_path):
            raise FileNotFoundError("❌ Image embeddings not found.")

        query = translate_query(query)
        image_embeddings = torch.load(self.image_embed_path, weights_only=True)
        text = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        results = []
        for name, img_emb in image_embeddings.items():
            similarity = torch.cosine_similarity(text_features, img_emb, dim=-1)
            results.append((name, similarity.item()))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ----------------------------------------- Search Image by Image ----------------------------------------- #
def search_by_image(self, image_path: str, top_k=6):
    if not os.path.exists(self.image_embed_path):
        raise FileNotFoundError("❌ Image embeddings not found.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

# ----------------------------------- Load precomputed image embeddings ----------------------------------- #
    image_embeddings = torch.load(self.image_embed_path, weights_only=True)  # dict[name] -> [1, D] CPU tensor

# -------------------------------- Encode the query image and L2-normalize -------------------------------- #
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load image: {e}")

    with torch.no_grad():
        img_in = self.preprocess(pil_img).unsqueeze(0).to(self.device)  # [1, C, H, W]
        q_feat = self.model.encode_image(img_in)                        # [1, D]
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)             # L2-normalize
        q_feat = q_feat.cpu()                                           # compare on CPU with stored tensors

# ---------------------------- Compute cosine similarity against all embeddings ---------------------------- #
    results = []
    basename = os.path.basename(image_path)
    for name, img_emb in image_embeddings.items():
        sim = torch.cosine_similarity(q_feat, img_emb, dim=-1).item()
        # Optional: skip exact self-match if the uploaded image exists in the dataset
        # if name == basename:
        #     continue
        results.append((name, sim))

    # 4) Sort by similarity and return Top-K
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

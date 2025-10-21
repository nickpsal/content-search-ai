from PIL import Image
import torch
import clip
from sentence_transformers import SentenceTransformer, util
import os

model = SentenceTransformer("./models/mclip_finetuned_coco_ready")
print("✅ Model loaded OK!")

base_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
finetuned_path = "./models/mclip_finetuned_coco_ready"
image_path = "./data/images/val2017/000000000139.jpg"


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
text_model = SentenceTransformer(finetuned_path, device=device)

image_dir = "./data/images/val2017"
query = "μια γάτα στο γρασίδι"
top_k = 5

# Load images
paths = [os.path.join(image_dir, p) for p in os.listdir(image_dir) if p.endswith(".jpg")][:500]

# Encode images
img_embs, imgs = [], []
for p in paths:
    img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    img_embs.append(emb.cpu())
    imgs.append(p)
img_embs = torch.cat(img_embs)

# Encode text
text_emb = text_model.encode(query, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
sims = util.cos_sim(text_emb, img_embs)[0]
top_results = torch.topk(sims, k=top_k)

# Create result grid
thumbs = [Image.open(imgs[i]).resize((256,256)) for i in top_results.indices]
grid_w = sum(t.width for t in thumbs)
grid = Image.new('RGB', (grid_w, thumbs[0].height))
x = 0
for t in thumbs:
    grid.paste(t, (x, 0))
    x += t.width
grid.save("results.jpg")
print("✅ Saved top results to results.jpg")
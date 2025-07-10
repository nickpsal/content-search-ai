import os
import json
import torch
import clip
from tqdm import tqdm

# Load CLIP model (ViT-B/32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to captions JSON file
captions_file = "../data/annotations/annotations/captions_val2017.json"
# Output path for text embeddings
output_file = "../data/embeddings/coco_val_text_embeddings.pt"

# Load the captions
with open(captions_file, 'r') as f:
    data = json.load(f)

# Dictionary to hold: {image_id: [list of embeddings]}
embeddings = {}

# Process each caption
for ann in tqdm(data['annotations'], desc="Extracting text embeddings"):
    caption = ann['caption']
    image_id = ann['image_id']
    image_name = f"{image_id:012}.jpg"

    # Tokenize and encode caption
    text = clip.tokenize(caption).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Store embedding (append to list in case of multiple captions per image)
    if image_name not in embeddings:
        embeddings[image_name] = []
    embeddings[image_name].append(text_features.cpu())

# Average multiple caption embeddings per image
final_embeddings = {}
for image_name, feats in embeddings.items():
    stacked = torch.stack(feats)
    avg_feat = stacked.mean(dim=0)
    final_embeddings[image_name] = avg_feat

# Save result
torch.save(final_embeddings, output_file)
print(f"\n✅ Saved {len(final_embeddings)} caption embeddings to {output_file}")

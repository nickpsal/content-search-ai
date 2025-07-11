import torch
import clip
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load precomputed image embeddings
image_embeddings = torch.load("../data/embeddings/coco_val_image_embeddings.pt")

# Ask user for a text query
query = input("🔍 Enter a text query: ")

# Tokenize and encode the query
with torch.no_grad():
    text_tokens = clip.tokenize([query]).to(device)
    text_embedding = model.encode_text(text_tokens)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

# Compute cosine similarity with all image embeddings
results = []
for filename, img_emb in image_embeddings.items():
    similarity = torch.cosine_similarity(text_embedding, img_emb, dim=-1)
    results.append((filename, similarity.item()))

# Sort results by similarity
results.sort(key=lambda x: x[1], reverse=True)

# Show top 5 most relevant images
top_k = 5
print(f"\nTop {top_k} results for: \"{query}\"")
for i in range(top_k):
    fname, score = results[i]
    print(f"{i + 1}. {fname} (score: {score:.4f})")

    # Load and display the image
    image_path = os.path.join("../data/images/val2017", fname)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"{fname} — Score: {score:.4f}")
    plt.axis("off")
    plt.show()

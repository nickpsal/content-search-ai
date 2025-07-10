import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Load CLIP model (ViT-B/32) and move it to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the COCO image folder
image_folder = "../data/images/val2017"

# Output file to save the image embeddings
embedding_file = "../data/embeddings/coco_val_image_embeddings.pt"

# List all image files
image_paths = [os.path.join(image_folder, fname)
               for fname in os.listdir(image_folder)
               if fname.endswith(".jpg")]

# Dictionary to store image embeddings
embeddings = {}

# Iterate through all images
for path in tqdm(image_paths, desc="Extracting image embeddings"):
    try:
        # Load and preprocess the image using CLIP's transforms
        image_pil: Image.Image = Image.open(path).convert("RGB")
        image = preprocess(image_pil).unsqueeze(0).to(device)

        # Extract the image features using CLIP
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize

        # Store the features with filename as key
        filename = os.path.basename(path)
        embeddings[filename] = image_features.cpu()

    except Exception as e:
        print(f"Error processing {path}: {e}")

# Save all embeddings to a .pt file
torch.save(embeddings, embedding_file)
print(f"\nSaved {len(embeddings)} image embeddings to {embedding_file}")
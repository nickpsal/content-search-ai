# scripts/download_coco_val_captions.py

import os
import requests
from tqdm import tqdm

# COCO captions (val2017) download URL
COCO_CAPTIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Target paths
annotations_dir = "../data/annotations"
zip_path = os.path.join(annotations_dir, "annotations_trainval2017.zip")
captions_json_path = os.path.join(annotations_dir, "annotations", "captions_val2017.json")

# Make sure folder exists
os.makedirs(annotations_dir, exist_ok=True)

def download_file(url, dest):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(dest)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))

def unzip_file(zip_file_path, extract_to):
    """Unzip zip file"""
    import zipfile
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Check if already downloaded
if not os.path.exists(captions_json_path):
    print("Downloading COCO captions annotations...")
    download_file(COCO_CAPTIONS_URL, zip_path)

    print("Extracting captions...")
    unzip_file(zip_path, annotations_dir)

    print("✅ Captions saved at:", captions_json_path)
else:
    print("✅ Captions already exist at:", captions_json_path)

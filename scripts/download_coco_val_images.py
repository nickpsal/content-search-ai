import os
import zipfile
import requests
from tqdm import tqdm

# Define the COCO val2017 URL
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"

# Set paths
data_dir = "../data/images"
zip_path = os.path.join(data_dir, "val2017.zip")
extract_dir = os.path.join(data_dir, "val2017")

# Create target folder if it doesn't exist
os.makedirs(data_dir, exist_ok=True)


def download_file(url, dest):
    """Download file from URL with progress bar"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(dest)}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))


def unzip_file(zip_file_path, extract_to):
    """Unzip zip file"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
    print("Downloading COCO val2017 images...")
    download_file(COCO_URL, zip_path)

    print("Extracting zip file...")
    unzip_file(zip_path, data_dir)

    print("✅ Done! Images extracted to:", extract_dir)
else:
    print("✅ Images already exist at:", extract_dir)

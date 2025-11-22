import os
import zipfile
import gdown
from tqdm import tqdm

class Model:
    def __init__(self, data_dir="models"):
        self.model_dir = os.path.abspath(data_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        # Google Drive ZIPs
        self.downloads = [
            {
                "id": "1U1pqD9g4_NwZLWW-Y_VYO2Qfk7ymzySN",
                "name": "mclip_finetuned_coco_ready.zip"
            },
            {
                "id": "1aDNpqCu1afl-7Cdn7UjCIp2x2iEU5jme",
                "name": "extra_model_files.zip"
            }
        ]

    def download_model(self):
        """Κατεβάζει και αποσυμπιέζει ΟΛΑ τα zip από Google Drive μέσα στο models/"""

        # Αν ο φάκελος έχει ήδη αρχεία → δεν ξανακατεβάζουμε
        existing = os.listdir(self.model_dir)
        if len(existing) > 0:
            print(f"✅ Model files already exist in {self.model_dir}")
            return

        # Loop για όλα τα ZIPs
        for item in self.downloads:
            zip_path = os.path.join(self.model_dir, item["name"])
            url = f"https://drive.google.com/uc?id={item['id']}"

            print(f"\n📥 Downloading: {item['name']}")
            gdown.download(url, zip_path, quiet=False, fuzzy=True)

            # Έλεγχος αν είναι zip
            if not zipfile.is_zipfile(zip_path):
                print(f"❌ Invalid ZIP: {item['name']}")
                continue

            # Αποσυμπίεση
            print(f"📦 Extracting {item['name']}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                for file in tqdm(z.namelist(), desc="Extracting", unit="file"):
                    z.extract(file, self.model_dir)

            # Σβήσε το zip
            os.remove(zip_path)

        print(f"\n✅ All model files extracted into {self.model_dir}")

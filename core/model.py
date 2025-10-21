import os
import zipfile
import gdown
from tqdm import tqdm

class Model:
    def __init__(self, model_id="1U1pqD9g4_NwZLWW-Y_VYO2Qfk7ymzySN", data_dir="models"):
        self.model_link_id = model_id
        self.model_dir = os.path.abspath(data_dir)
        self.model_zip = os.path.join(self.model_dir, "mclip_finetuned_coco_ready.zip")

    def download_model(self):
        """Κατεβάζει και αποσυμπιέζει το μοντέλο από Google Drive μέσα στο models/"""
        os.makedirs(self.model_dir, exist_ok=True)

        # Αν υπάρχουν ήδη αρχεία στο models/, δεν ξανακατεβάζουμε
        if len(os.listdir(self.model_dir)) > 0:
            print(f"✅ Model already exists in {self.model_dir}")
            return

        # Direct Google Drive URL
        url = f"https://drive.google.com/uc?id={self.model_link_id}"

        print(f"\n📥 Downloading model from Google Drive...")
        gdown.download(url, self.model_zip, quiet=False, fuzzy=True)

        # Έλεγχος αν είναι έγκυρο zip
        if not zipfile.is_zipfile(self.model_zip):
            print("❌ Downloaded file is not a valid ZIP. Check the Google Drive link ID.")
            return

        # 📦 Αποσυμπίεση απευθείας στο models/
        with zipfile.ZipFile(self.model_zip, 'r') as zip_ref:
            files = zip_ref.namelist()
            print(f"\n📦 Extracting {len(files)} files into {self.model_dir}...")
            for file in tqdm(files, desc="Extracting", unit="file"):
                zip_ref.extract(file, self.model_dir)

        os.remove(self.model_zip)
        print(f"✅ Model extracted successfully into {self.model_dir}")

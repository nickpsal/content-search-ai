import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pathlib import Path
import numpy as np
from PIL import Image
import torch

# --- Model + DB ---
from core.image_search import ImageSearcher
from core.db.database_helper import DatabaseHelper


# ============================================
# 🔧 Helper: Get correct relative DB path
# ============================================
def make_relative(full_path: str) -> str:
    """
    Μετατρέπει absolute path σε:
    data/images/other/filename.jpg
    """
    full_path = full_path.replace("\\", "/")

    if "/data/" in full_path:
        rel = full_path.split("/data/")[1]
        return f"data/{rel}"

    print("❌ Could not compute relative path for:", full_path)
    return None


# ============================================
# 📌 Watchdog Handler
# ============================================
class ImageFolderHandler(FileSystemEventHandler):

    def __init__(self):
        # ------------------------
        # BASE PATH
        # ------------------------
        self.base_dir = Path(__file__).resolve().parents[2]

        # ------------------------
        # DB
        # ------------------------
        db_path = self.base_dir / "content_search_ai.db"
        print("📌 Initializing DB helper…")
        self.db = DatabaseHelper(str(db_path))

        #initialise Database
        self.db.initialise_database()

        # ------------------------
        # Model
        # ------------------------
        print("📌 Initializing ML model for Watchdog…")
        self.searcher = ImageSearcher()  # <--- FIXED
        self.image_model = self.searcher.image_model
        self.preprocess = self.searcher.preprocess
        self.device = self.searcher.device

        self.watch_dir = str(self.base_dir / "data/images/other")

    # --------------------------------------------
    # 🗑️ FILE DELETED
    # --------------------------------------------
    def on_deleted(self, event):
        if event.is_directory:
            return

        rel_path = make_relative(event.src_path)
        if not rel_path:
            return

        print(f"🗑️ Deleted → {rel_path}")
        self.db.delete_image(rel_path)

    # --------------------------------------------
    # 🆕 FILE CREATED
    # --------------------------------------------
    def on_created(self, event):
        if event.is_directory:
            return

        full_path = event.src_path.replace("\\", "/")
        rel_path = make_relative(full_path)
        if not rel_path:
            return

        print(f"🆕 Created → {rel_path}")

        try:
            # Load + resize image
            img = Image.open(full_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            # Extract embedding (normalized)
            with torch.no_grad():
                emb = self.image_model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            emb_bytes = emb.cpu().numpy().astype(np.float32).tobytes()

            filename = os.path.basename(full_path)

            # Insert into DB
            self.db.insert_image(filename, rel_path, emb_bytes)
            print(f"💾 Inserted into DB → {filename}")

        except Exception as e:
            print(f"❌ Error processing new image {full_path}: {e}")


# ============================================
# 🚀 Start Watchdog
# ============================================
def start_watch():
    handler = ImageFolderHandler()
    observer = Observer()

    watch_dir = handler.watch_dir

    print("👀 Watching folder:")
    print(watch_dir)

    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

# core/watchdog/watch_pdfs.py
import time
import numpy as np
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core.db.database_helper import DatabaseHelper
from core.pdf_search import PDFSearcher


# ============================================
# 🔧 Helper – Convert absolute → relative
# ============================================
def make_relative(full_path: str) -> str:
    full_path = full_path.replace("\\", "/")

    if "/data/" in full_path:
        rel = full_path.split("/data/")[1]
        return f"data/{rel}"

    print("❌ Could not compute relative path for:", full_path)
    return None


# ============================================
# 📘 PDF WATCHDOG HANDLER
# ============================================
class PDFHandler(FileSystemEventHandler):

    def __init__(self):
        # ===================================
        # 📌 CORRECT BASE DIR
        # core/watchdog/watch_pdfs.py
        # parents[2] → content-search-ai
        # ===================================
        self.base_dir = Path(__file__).resolve().parents[2]

        print("📌 PDF Watchdog BASE DIR =", self.base_dir)

        # -------------------------
        # DB
        # -------------------------
        self.db_path = self.base_dir / "content_search_ai.db"
        print("📌 DB Path =", self.db_path)
        self.db = DatabaseHelper(str(self.db_path))

        # -------------------------
        # PDF Search Model
        # -------------------------
        model_path = self.base_dir / "models/mclip_finetuned_coco_ready"
        print("📌 Loading PDF M-CLIP model from:", model_path)

        self.pdf_searcher = PDFSearcher(
            db_path=str(self.db_path),
            model_path=str(model_path)
        )

        # -------------------------
        # FOLDER TO WATCH
        # -------------------------
        self.watch_dir = str(self.base_dir / "data/pdfs")


    # -------------------------------------------------
    # 🗑️ DELETE
    # -------------------------------------------------
    def on_deleted(self, event):
        if event.is_directory:
            return

        rel_path = make_relative(event.src_path)
        if not rel_path:
            return

        print(f"🗑️ PDF Deleted → {rel_path}")
        self.db.delete_pdf(rel_path)


    # -------------------------------------------------
    # 🆕 CREATE
    # -------------------------------------------------
    def on_created(self, event):
        if event.is_directory:
            return

        full_path = event.src_path.replace("\\", "/")
        rel_path = make_relative(full_path)
        if not rel_path:
            return

        print(f"🆕 PDF Created → {rel_path}")

        try:
            pages = self.pdf_searcher.get_pdf_pages_embeddings(full_path)

            if not pages:
                print("⚠️ No valid pages extracted.")
                return

            for page_num, emb, text in pages:
                emb_bytes = emb.astype(np.float32).tobytes()

                self.db.insert_pdf_page(
                    rel_path,
                    page_num,
                    text,
                    emb_bytes
                )

            print(f"💾 Inserted {len(pages)} pages → {rel_path}")

        except Exception as e:
            print(f"❌ Error processing PDF {full_path}: {e}")


# ============================================
# 🚀 START WATCHDOG
# ============================================
def start_watch():
    handler = PDFHandler()
    observer = Observer()

    watch_dir = handler.watch_dir
    print("\n📄 Watching PDF folder:")
    print(watch_dir)

    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

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
def make_relative(full_path: str) -> str | None:
    full_path = full_path.replace("\\", "/")
    if "/data/" in full_path:
        return "data/" + full_path.split("/data/")[1]
    return None


# ============================================
# 📘 PDF WATCHDOG HANDLER
# ============================================
class PDFHandler(FileSystemEventHandler):

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.db_path = self.base_dir / "content_search_ai.db"
        self.db = DatabaseHelper(str(self.db_path))

        model_path = self.base_dir / "models/mclip_finetuned_coco_ready"
        self.pdf_searcher = PDFSearcher(
            db_path=str(self.db_path),
            model_path=str(model_path)
        )

        self.watch_dir = str(self.base_dir / "data/pdfs")

        # 🧠 debounce state
        self._last_processed: dict[str, float] = {}

        print("📄 PDF Watchdog ready")
        print("📌 Watching:", self.watch_dir)


    # -------------------------------------------------
    # ⏳ Wait until file is stable & readable
    # -------------------------------------------------
    def _wait_for_ready(self, path: str, timeout: int = 20) -> bool:
        last_size = -1
        start = time.time()

        while time.time() - start < timeout:
            try:
                size = Path(path).stat().st_size
                if size == last_size and size > 0:
                    return True
                last_size = size
            except FileNotFoundError:
                return False

            time.sleep(0.5)

        return False


    # -------------------------------------------------
    # 🆕 CREATE / MODIFY (same logic)
    # -------------------------------------------------
    def on_created(self, event):
        self._handle_event(event)

    def on_modified(self, event):
        self._handle_event(event)


    # -------------------------------------------------
    # 🧠 Unified handler
    # -------------------------------------------------
    def _handle_event(self, event):
        if event.is_directory:
            return

        if not event.src_path.lower().endswith(".pdf"):
            return

        full_path = event.src_path.replace("\\", "/")
        rel_path = make_relative(full_path)
        if not rel_path:
            return

        now = time.time()

        # 🛑 Debounce (avoid duplicate triggers)
        last = self._last_processed.get(full_path, 0)
        if now - last < 2:
            return
        self._last_processed[full_path] = now

        print(f"📄 PDF change detected → {rel_path}")

        # ⏳ Wait for full write
        if not self._wait_for_ready(full_path):
            print(f"❌ PDF not ready (timeout) → {rel_path}")
            return

        try:
            # 🧹 Clean old pages first
            self.db.delete_pdf(rel_path)

            pages = self.pdf_searcher.get_pdf_pages_embeddings(full_path)
            if not pages:
                print(f"⚠️ No valid pages → {rel_path}")
                return

            for page_num, emb, text in pages:
                emb_bytes = emb.astype(np.float32).tobytes()
                self.db.insert_pdf_page(
                    rel_path,
                    page_num,
                    text,
                    emb_bytes
                )

            print(f"💾 Indexed {len(pages)} pages → {rel_path}")

        except Exception as e:
            print(f"❌ PDF processing error {rel_path}: {e}")


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


# ============================================
# 🚀 START WATCHDOG
# ============================================
def start_watch():
    handler = PDFHandler()
    observer = Observer()

    observer.schedule(handler, handler.watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

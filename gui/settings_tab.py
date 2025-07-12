import os
import ttkbootstrap as tb
from tkinter import ttk
from core import ImageSearcher

def create_settings_tab(notebook):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Settings")

    searcher = ImageSearcher(data_dir="data")

    def check_data_exists():
        results = []
        if os.path.exists(searcher.image_dir):
            results.append("🖼️ Images: ✅")
        else:
            results.append("🖼️ Images: ❌")
        if os.path.exists(searcher.caption_file):
            results.append("📝 Captions: ✅")
        else:
            results.append("📝 Captions: ❌")
        if os.path.exists(searcher.image_embed_path):
            results.append("📦 Image Embeddings: ✅")
        else:
            results.append("📦 Image Embeddings: ❌")
        if os.path.exists(searcher.text_embed_path):
            results.append("🧠 Text Embeddings: ✅")
        else:
            results.append("🧠 Text Embeddings: ❌")
        return "\n".join(results)

    status_label = tb.Label(tab, text="Έλεγχος δεδομένων...", font=("Segoe UI", 10), justify="left")
    status_label.pack(pady=(10, 5))

    def handle_download():
        searcher.download_coco_data()
        searcher.extract_image_embeddings()
        searcher.extract_text_embeddings()
        status_label.config(text=check_data_exists())

    download_btn = tb.Button(tab, text="⬇️ Download & Extract All Files", command=handle_download)
    download_btn.pack(pady=5)

    status_label.config(text=check_data_exists())

    return tab

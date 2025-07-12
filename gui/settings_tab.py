import os
import sys

import ttkbootstrap as tb
from tkinter import ttk, BooleanVar, Toplevel
from core import ImageSearcher, TextRedirector

def create_settings_tab(notebook):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Settings")

    searcher = ImageSearcher(data_dir="../data")
    data_exists = BooleanVar(value=False)

    # Status label
    status_label = tb.Label(tab, text="Έλεγχος δεδομένων...", font=("Segoe UI", 10), justify="left")
    status_label.pack(pady=(10, 5))

    # -------- Έλεγχος Δεδομένων --------
    def check_data_exists():
        all_ok = True
        results = []

        if os.path.exists(searcher.image_dir):
            results.append("🖼️ Images: ✅")
        else:
            results.append("🖼️ Images: ❌")
            all_ok = False

        if os.path.exists(searcher.caption_file):
            results.append("📝 Captions: ✅")
        else:
            results.append("📝 Captions: ❌")
            all_ok = False

        if os.path.exists(searcher.image_embed_path):
            results.append("📦 Image Embeddings: ✅")
        else:
            results.append("📦 Image Embeddings: ❌")
            all_ok = False

        if os.path.exists(searcher.text_embed_path):
            results.append("🧠 Text Embeddings: ✅")
        else:
            results.append("🧠 Text Embeddings: ❌")
            all_ok = False

        data_exists.set(all_ok)
        return "\n".join(results)

    # -------- Modal & Download Handler --------
    def handle_download():
        modal = Toplevel(tab)
        modal.title("Λήψη Δεδομένων")
        modal.geometry("600x400")
        modal.resizable(False, False)
        modal.grab_set()  # κάνε το modal

        # Progress bar
        progress = tb.Progressbar(modal, mode="determinate", maximum=3)
        progress.pack(fill="x", padx=10, pady=(10, 5))
        progress["value"] = 0

        # Text output
        output = tb.ScrolledText(modal, font=("Consolas", 9), height=15)
        output.pack(fill="both", expand=True, padx=10, pady=10)
        sys.stdout = TextRedirector(output)

        # Εκτέλεση σε ξεχωρισμένο thread (ώστε να μην παγώνει το GUI)
        def run_download():
            try:
                print("🚀 Ξεκινά η λήψη και η επεξεργασία...\n")
                searcher.download_coco_data()
                progress["value"] += 1
                progress.update_idletasks()

                searcher.extract_image_embeddings()
                progress["value"] += 1
                progress.update_idletasks()

                searcher.extract_text_embeddings()
                progress["value"] += 1
                progress.update_idletasks()

                print("\n✅ Όλα τα δεδομένα ετοιμάστηκαν.")
            except Exception as e:
                print(f"\n❌ Σφάλμα: {e}")
            finally:
                progress.stop()
                sys.stdout = sys.__stdout__
                status_label.config(text=check_data_exists())
                if data_exists.get():
                    download_btn.config(state="disabled")
                modal.destroy()  # ✅ κλείσιμο modal αυτόματα

        import threading
        threading.Thread(target=run_download).start()

    # -------- Κουμπί Λήψης --------
    download_btn = tb.Button(tab, text="⬇️ Download & Extract All Files", command=handle_download)
    download_btn.pack(pady=5)

    # Αρχικός έλεγχος
    status_label.config(text=check_data_exists())
    if data_exists.get():
        download_btn.config(state="disabled")

    return tab
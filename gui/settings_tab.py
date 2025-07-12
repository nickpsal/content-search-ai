import os
import sys
import threading
import ttkbootstrap as tb
from tkinter import ttk, BooleanVar, Toplevel
from core import ImageSearcher, TextRedirector


def create_settings_tab(notebook):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Settings")

    searcher = ImageSearcher(data_dir="../data")
    data_exists = BooleanVar(value=False)

    # -------- LabelFrame: Κατάσταση Δεδομένων --------
    status_frame = ttk.LabelFrame(tab, text="📦 Κατάσταση Δεδομένων")
    status_frame.pack(pady=(20, 10), padx=20, fill="x")

    status_labels = {}

    def update_status(name, emj, ok):
        text = f"{emj} {name}:"
        status_labels[name].config(text=text)
        check = "✅" if ok else "❌"
        status_labels[name + "_val"].config(text=check)

    def check_data_exists():
        all_ok = True
        items = {
            "Images": (searcher.image_dir, "🖼️"),
            "Captions": (searcher.caption_file, "📝"),
            "Image Embeddings": (searcher.image_embed_path, "📦"),
            "Text Embeddings": (searcher.text_embed_path, "🧠"),
        }

        for i, (name, (path, emoji_icon)) in enumerate(items.items()):
            ok = os.path.exists(path)
            update_status(name, emoji_icon, ok)
            if not ok:
                all_ok = False

        data_exists.set(all_ok)

    # Δημιουργία γραμμών εμφάνισης status
    row = 0
    for label, (_, emoji) in {
        "Images": ("", "🖼️"),
        "Captions": ("", "📝"),
        "Image Embeddings": ("", "📦"),
        "Text Embeddings": ("", "🧠"),
    }.items():
        l1 = tb.Label(status_frame, text="", anchor="w", font=("Segoe UI", 10))
        l2 = tb.Label(status_frame, text="", anchor="center", font=("Segoe UI", 10, "bold"))
        l1.grid(row=row, column=0, sticky="w", padx=10, pady=3)
        l2.grid(row=row, column=1, sticky="e", padx=10)
        status_labels[label] = l1
        status_labels[label + "_val"] = l2
        row += 1

    # -------- Modal & Download Handler --------
    def handle_download():
        modal = Toplevel(tab)
        modal.title("Λήψη Δεδομένων")
        modal.geometry("600x400")
        modal.resizable(False, False)
        modal.grab_set()

        progress = tb.Progressbar(modal, mode="determinate", maximum=3)
        progress.pack(fill="x", padx=10, pady=(10, 5))

        output = tb.ScrolledText(modal, font=("Consolas", 9), height=15)
        output.pack(fill="both", expand=True, padx=10, pady=10)
        sys.stdout = TextRedirector(output)

        def run_download():
            try:
                print("🚀 Ξεκινά η λήψη και η επεξεργασία...\n")
                searcher.download_coco_data()
                progress["value"] = 1
                progress.update_idletasks()

                searcher.extract_image_embeddings()
                progress["value"] = 2
                progress.update_idletasks()

                searcher.extract_text_embeddings()
                progress["value"] = 3
                progress.update_idletasks()

                print("\n✅ Όλα τα δεδομένα ετοιμάστηκαν.")
            except Exception as e:
                print(f"\n❌ Σφάλμα: {e}")
            finally:
                sys.stdout = sys.__stdout__
                check_data_exists()
                if data_exists.get():
                    download_btn.config(state="disabled")
                modal.destroy()

        threading.Thread(target=run_download).start()

    # -------- Κουμπί Λήψης --------
    download_btn = tb.Button(tab, text="⬇️ Download & Extract All Files", command=handle_download)
    download_btn.pack(pady=5)

    check_data_exists()
    if data_exists.get():
        download_btn.config(state="disabled")

    return tab

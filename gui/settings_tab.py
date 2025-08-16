import os
import threading
import ttkbootstrap as tb
from tkinter import ttk, BooleanVar, Toplevel

from core import ImageSearcher

def create_settings_tab(notebook):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Settings")

    style = tb.Style()
    searcher = ImageSearcher(data_dir="../data")
    data_exists = BooleanVar(master=tab, value=False)

    # -------------------------- LabelFrame: Κατάσταση Δεδομένων ----------------------------#
    status_frame = tb.Labelframe(tab, text="📦 Κατάσταση Δεδομένων", padding=10)
    status_frame.pack(pady=(20, 10), padx=20, fill="x")

    # ωραία στοίχιση των δύο στηλών
    status_frame.grid_columnconfigure(0, weight=1)
    status_frame.grid_columnconfigure(1, weight=0)

    status_labels = {}

    def update_status(name, emj, ok):
        # Καλείται πάντα στο main thread
        text = f"{emj} {name}:"
        status_labels[name].config(text=text)
        status_labels[name + "_val"].config(text=("✅" if ok else "❌"))

    def check_data_exists():
        all_ok = True
        items = [
            ("Images",             (searcher.image_dir,        "🖼️"), True),   # dir
            ("Captions",           (searcher.caption_file,     "📝"), False),  # file
            ("Image Embeddings",   (searcher.image_embed_path, "📦"), False),
            ("Text Embeddings",    (searcher.text_embed_path,  "🧠"), False),
        ]
        for name, (path, emoji_icon), is_dir in items:
            ok = os.path.isdir(path) if is_dir else os.path.exists(path)
            update_status(name, emoji_icon, ok)
            if not ok:
                all_ok = False
        data_exists.set(all_ok)

    # Δημιουργία γραμμών εμφάνισης status
    row = 0
    for label, _emoji in [
        ("Images", "🖼️"),
        ("Captions", "📝"),
        ("Image Embeddings", "📦"),
        ("Text Embeddings", "🧠"),
    ]:
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
        modal.geometry("640x420")
        modal.resizable(False, False)
        modal.grab_set()

        progress = tb.Progressbar(modal, mode="determinate", maximum=3)
        progress.pack(fill="x", padx=10, pady=(10, 5))

        output = tb.ScrolledText(modal, font=("Consolas", 9), height=15)
        output.pack(fill="both", expand=True, padx=10, pady=10)

        # Thread-safe helpers (όλες οι UI ενημερώσεις μέσω after)
        def set_progress(value):
            modal.after(0, lambda: (progress.config(value=value), progress.update_idletasks()))

        def log(text):
            def _append():
                output.insert("end", text)
                output.see("end")
            modal.after(0, _append)

        # Απενεργοποίηση κουμπιού όσο «τρέχει»
        download_btn.config(state="disabled")

        def run_download():
            try:
                log("🚀 Ξεκινά η λήψη και η επεξεργασία...\n")
                searcher.download_coco_data()
                set_progress(1)

                log("📦 Εξαγωγή image embeddings...\n")
                searcher.extract_image_embeddings()
                set_progress(2)

                log("🧠 Εξαγωγή text embeddings...\n")
                searcher.extract_text_embeddings()
                set_progress(3)

                log("\n✅ Όλα τα δεδομένα ετοιμάστηκαν.\n")
            except Exception as e:
                log(f"\n❌ Σφάλμα: {e}\n")
                # αν αποτύχει, άφησε ξανά ενεργό το κουμπί
                tab.after(0, lambda: download_btn.config(state="normal"))
            finally:
                # Ανανέωση status στο main thread
                tab.after(0, check_data_exists)
                # Κλείσιμο modal μετά από λίγο ώστε να δει ο χρήστης το τελικό log
                modal.after(500, modal.destroy)

        threading.Thread(target=run_download, daemon=True).start()

    # -------- Κουμπί Λήψης --------
    download_btn = tb.Button(tab, text="⬇️ Download & Extract All Files", command=handle_download)
    download_btn.pack(pady=16)

    # --------------------------- LabelFrame: Αλλαγή Background -----------------------------#
    color_theme_frame = tb.Labelframe(tab, text="🎨 Αλλαγή Background Theme", padding=10)
    color_theme_frame.pack(pady=(10, 20), padx=20, fill="x")

    tb.Label(color_theme_frame, text="Επίλεξε θέμα:").pack(anchor="w", padx=2)

    available_themes = sorted(style.theme_names())
    current_theme = style.theme.name
    selected_theme = tb.StringVar(value=current_theme)

    theme_row = tb.Frame(color_theme_frame)
    theme_row.pack(fill="x", pady=(6, 0))

    theme_combo = tb.Combobox(
        theme_row,
        textvariable=selected_theme,
        values=available_themes,
        state="readonly",
        width=22,
    )
    theme_combo.pack(side="left")

    def apply_theme():
        th = selected_theme.get()
        style.theme_use(th)

    tb.Button(theme_row, text="Apply", command=apply_theme).pack(side="left", padx=8)

    tb.Label(
        color_theme_frame,
        text="Tip: Dark themes → darkly, cyborg, superhero, vapor.",
    ).pack(anchor="w", pady=(8, 0))

    # Αρχικός έλεγχος δεδομένων & disable κουμπιού αν όλα υπάρχουν
    check_data_exists()
    if data_exists.get():
        download_btn.config(state="disabled")

    return tab

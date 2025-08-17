import os
from tkinter import ttk, filedialog
import ttkbootstrap as tb
from PIL import Image, ImageTk
from core import ImageSearcher
from typing import Optional, cast


# -------------------------------------- Images Tab ---------------------------------------- #
def create_image_tab(notebook):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Image")

    searcher = ImageSearcher(data_dir="data")

    # -------------------------------------- Main Content ----------------------------------- #
    content_frame = ttk.Frame(tab)
    content_frame.pack(pady=10, fill="x")

    # -------------------------------------- Text Query Section ----------------------------- #
    text_group = ttk.LabelFrame(content_frame, text="Text-to-Image Search")
    text_group.pack(padx=10, pady=10, fill="x")

    query_label = tb.Label(text_group, text="🔎 Εισάγετε ερώτημα:")
    query_label.pack(pady=(10, 0), anchor="center")

    query_entry = tb.Entry(text_group, width=50)
    query_entry.pack(pady=(0, 10), anchor="center")

    search_btn = tb.Button(text_group, text="🔍 Search")
    search_btn.pack(pady=(0, 10), anchor="center")

    # -------------------------------------- Image Query Section ---------------------------- #
    image_group = ttk.LabelFrame(content_frame, text="Image-to-Image Search (Upload)")
    image_group.pack(padx=10, pady=10, fill="x")

    selected_image_path = {"path": None}
    preview_label = tb.Label(image_group, text="(Δεν έχει επιλεγεί εικόνα)")
    preview_label.pack(pady=(8, 6), anchor="center")

    pick_btn = tb.Button(image_group, text="📁 Επιλογή εικόνας")
    pick_btn.pack(pady=(0, 6), anchor="center")

    img_search_btn = tb.Button(image_group, text="🖼️ Find Similar")
    img_search_btn.pack(pady=(0, 10), anchor="center")

    # -------------------------------------- Results Section -------------------------------- #
    result_label = tb.Label(content_frame, text="", font=("Segoe UI", 9))
    result_label.pack(pady=(0, 10), anchor="center")

    results_frame = ttk.Frame(content_frame)
    results_frame.pack(pady=10, fill="x")

    image_labels = []

    def clear_results():
        for label in image_labels:
            label.destroy()
        image_labels.clear()

    def display_results(results, img_size=(200, 200)):
        clear_results()

        if not results:
            result_label.config(text="❌ Δεν βρέθηκαν αποτελέσματα.")
            return

        result_label.config(text=f"📸 Top {len(results)} αποτελέσματα:")

        for i, (name, score) in enumerate(results):
            img_path = os.path.join(searcher.image_dir, name)
            try:
                img = Image.open(img_path).resize(img_size)
            except Exception:
                continue

            tk_img = ImageTk.PhotoImage(img)

            img_label = tb.Label(
                results_frame,
                image=tk_img,  # type: ignore[arg-type]
                text=f"{name}\nScore: {score:.4f}",
                compound="top",
                anchor="center"
            )
            img_label.image = tk_img

            row = i // 3
            col = i % 3
            img_label.grid(row=row, column=col, padx=10, pady=10, sticky="n")

            image_labels.append(img_label)

    # -------------------------------------- Actions ---------------------------------------- #
    def run_search_text():
        query = query_entry.get().strip()
        if not query:
            result_label.config(text="⚠️ Πληκτρολόγησε κάτι πρώτα.")
            return
        try:
            result_label.config(text="⏳ Αναζήτηση...")
            tab.update_idletasks()
            results = searcher.search_by_query(query, top_k=6)
            display_results(results, img_size=(150, 150))
        except Exception as e:
            result_label.config(text=f"❌ Σφάλμα: {e}")

    def choose_image():
        path = filedialog.askopenfilename(
            title="Επιλογή εικόνας",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        selected_image_path["path"] = path
        try:
            img = Image.open(path).copy()
            img.thumbnail((240, 240))
            tk_img = ImageTk.PhotoImage(img)
            preview_label.configure(image=tk_img, compound="bottom", text=os.path.basename(path))
            preview_label.image = tk_img
        except Exception as e:
            preview_label.configure(text=f"❌ Αποτυχία φόρτωσης προεπισκόπησης: {e}", image="")
            preview_label.image = None

    def run_search_image():
        p: Optional[str] = selected_image_path.get("path")
        if not p:
            result_label.config(text="⚠️ Επίλεξε πρώτα μία εικόνα.")
            return

        path = cast(str, p)

        try:
            result_label.config(text="⏳ Αναζήτηση παρόμοιων...")
            tab.update_idletasks()
            results = searcher.search_by_image(path, top_k=6)
            display_results(results, img_size=(150, 150))
        except Exception as e:
            result_label.config(text=f"❌ Σφάλμα: {e}")

    # -------------------------------------- Bind Buttons ----------------------------------- #
    search_btn.config(command=run_search_text)
    pick_btn.config(command=choose_image)
    img_search_btn.config(command=run_search_image)

    return tab

import os
from tkinter import ttk
import ttkbootstrap as tb
from PIL import Image, ImageTk
from core import ImageSearcher


def create_image_tab(notebook):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Image")

    searcher = ImageSearcher(data_dir="../data")

    # Κύριο περιεχόμενο
    content_frame = ttk.Frame(tab)
    content_frame.pack(pady=10)

    query_label = tb.Label(content_frame, text="🔎 Εισάγετε ερώτημα:")
    query_label.pack(pady=(10, 0), anchor="center")

    query_entry = tb.Entry(content_frame, width=50)
    query_entry.pack(pady=(0, 10), anchor="center")

    search_btn = tb.Button(content_frame, text="🔍 Search")
    search_btn.pack(pady=(0, 10), anchor="center")

    result_label = tb.Label(content_frame, text="", font=("Segoe UI", 9))
    result_label.pack(pady=(0, 10), anchor="center")

    results_frame = ttk.Frame(content_frame)
    results_frame.pack(pady=10, fill="x")

    image_labels = []

    def display_results(results, img_size=(200, 200)):
        for label in image_labels:
            label.destroy()
        image_labels.clear()

        if not results:
            result_label.config(text="❌ Δεν βρέθηκαν αποτελέσματα.")
            return

        result_label.config(text=f"📸 Top {len(results)} αποτελέσματα:")

        for i, (name, score) in enumerate(results):
            img_path = os.path.join(searcher.image_dir, name)
            img = Image.open(img_path).resize(img_size)
            tk_img = ImageTk.PhotoImage(img)

            img_label = tb.Label(results_frame, image=tk_img,  # type: ignore[arg-type]
                                 text=f"{name}\nScore: {score:.4f}",
                                 compound="top", anchor="center")
            img_label.image = tk_img

            row = i // 3
            col = i % 3
            img_label.grid(row=row, column=col, padx=10, pady=10, sticky="n")

            image_labels.append(img_label)

    def run_search():
        query = query_entry.get().strip()
        if not query:
            result_label.config(text="⚠️ Πληκτρολόγησε κάτι πρώτα.")
            return

        try:
            result_label.config(text="⏳ Αναζήτηση...")
            tab.update_idletasks()

            results = searcher.search(query, top_k=6)
            display_results(results, img_size=(150, 150))
        except Exception as e:
            result_label.config(text=f"❌ Σφάλμα: {e}")

    # Συνδέουμε το κουμπί
    search_btn.config(command=run_search)

    return tab

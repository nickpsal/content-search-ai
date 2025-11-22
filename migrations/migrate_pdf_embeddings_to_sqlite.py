import os
import sys
import sqlite3
import numpy as np
import torch
from tqdm import tqdm

# ===========================================
# FIX PYTHON PATH SO WE CAN IMPORT core/*
# ===========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from core.pdf_search import PDFSearcher
# ===========================================
# CONFIG
# ===========================================
DB_PATH = os.path.join(BASE_DIR, "content_search_ai.db")
PDF_FOLDER = os.path.join(BASE_DIR, "data", "pdfs")

# ===========================================
# INSERT
# ===========================================
def insert_pdf_page(cursor, pdf_path, page_num, text, emb_blob):
    cursor.execute(
        """
        INSERT INTO pdf_pages (pdf_path, page_number, text_content, embedding)
        VALUES (?, ?, ?, ?)
        """,
        (pdf_path, page_num, text, emb_blob)
    )

# ===========================================
# MAIN MIGRATION
# ===========================================
def migrate_pdfs():
    print(f"📁 Base dir: {BASE_DIR}")
    print(f"🗂️ Database: {DB_PATH}")
    print(f"📄 PDF folder: {PDF_FOLDER}")

    # load searcher (loads M-CLIP)
    searcher = PDFSearcher(model_path=os.path.join(BASE_DIR, "models", "mclip_finetuned_coco_ready"))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    print(f"🔍 Found {len(pdf_files)} PDFs to process")

    for file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_FOLDER, file)

        # absolute + normalized path
        abs_path = os.path.abspath(pdf_path).replace("\\", "/")

        pages = searcher.get_pdf_pages_embeddings(pdf_path)
        if not pages:
            print(f"⚠️ No valid pages in {file}")
            continue

        print(f"   → {file}: {len(pages)} valid pages")

        for page_num, emb, text in pages:
            # convert tensor → numpy → bytes
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

            emb_blob = emb.astype(np.float32).tobytes()

            insert_pdf_page(cursor, abs_path, page_num, text, emb_blob)

        conn.commit()

    conn.close()
    print("\n🎉 PDF migration completed successfully!")


# ===========================================
# RUN
# ===========================================
if __name__ == "__main__":
    migrate_pdfs()

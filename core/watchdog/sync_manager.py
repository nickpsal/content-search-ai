import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from core.db.database_helper import DatabaseHelper
from core.image_search import ImageSearcher
from core.pdf_search import PDFSearcher
from core.emotion_model_v5 import EmotionModelV5
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "content_search_ai.db"
db = DatabaseHelper(str(DB_PATH))


# ---------------------------------------------------
# 🟦 IMAGE SYNC
# ---------------------------------------------------
def sync_images():
    print("🔵 Syncing images...")

    searcher = ImageSearcher()
    model = searcher.image_model
    preprocess = searcher.preprocess
    device = searcher.device

    images_dir = BASE_DIR / "data/images"
    fs_files = {f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}

    db_files = set(db.get_all_image_paths())

    # A: Missing in database → INSERT
    to_insert = fs_files - db_files
    # B: Missing in folder → DELETE
    to_delete = db_files - fs_files

    for filename in to_insert:
        full_path = images_dir / filename
        rel_path = f"data/images/{filename}"

        try:
            img = Image.open(full_path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            emb_bytes = emb.cpu().numpy().astype(np.float32).tobytes()
            db.insert_image(filename, rel_path, emb_bytes)
            print(f"   ➕ Inserted image: {filename}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    for filename in to_delete:
        rel_path = f"data/images/{filename}"
        db.delete_image(rel_path)
        print(f"   ➖ Removed missing image: {filename}")


# ---------------------------------------------------
# 🟪 PDF SYNC
# ---------------------------------------------------
def sync_pdfs():
    print("🟣 Syncing PDFs...")

    pdf_dir = BASE_DIR / "data/pdfs"
    fs_files = {f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")}
    db_files = set(db.get_all_pdf_paths())

    # Clean DB paths for JUST filenames
    db_files = {Path(p).name for p in db_files}

    to_insert = fs_files - db_files
    to_delete = db_files - fs_files

    searcher = PDFSearcher(db_path=str(DB_PATH), model_path=str(BASE_DIR / "models/mclip_finetuned_coco_ready"))

    for filename in to_insert:
        full_path = pdf_dir / filename
        rel_path = f"data/pdfs/{filename}"

        try:
            pages = searcher.get_pdf_pages_embeddings(str(full_path))
            for pageno, emb, text in pages:
                db.insert_pdf_page(rel_path, pageno, text, emb.astype(np.float32).tobytes())
            print(f"   ➕ Inserted PDF: {filename}")
        except Exception as e:
            print(f"   ❌ Error PDF: {e}")

    for filename in to_delete:
        rel_path = f"data/pdfs/{filename}"
        db.delete_pdf(rel_path)
        print(f"   ➖ Removed missing PDF: {filename}")


# ---------------------------------------------------
# 🟥 AUDIO SYNC
# ---------------------------------------------------
def sync_audio():
    print("🟥 Syncing audio...")

    audio_dir = BASE_DIR / "data/audio"
    fs_files = {f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")}
    db_files = set(db.get_all_audio_paths())
    db_files = {Path(p).name for p in db_files}

    to_insert = fs_files - db_files
    to_delete = db_files - fs_files

    # Load models once
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    emotion_model = EmotionModelV5(ckpt_path=str(BASE_DIR / "models/best_model_audio_emotion_v5.pt"))
    mclip = SentenceTransformer(str(BASE_DIR / "models/mclip_finetuned_coco_ready"))

    for filename in to_insert:
        full_path = audio_dir / filename
        rel_path = f"data/audio/{filename}"

        try:
            segments, _ = whisper.transcribe(str(full_path), beam_size=1)
            transcript = " ".join(seg.text for seg in segments).strip()

            emb = mclip.encode(transcript, normalize_embeddings=True).astype(np.float32)
            emb_bytes = emb.tobytes()

            emotion, probs = emotion_model.predict(str(full_path))

            db.insert_audio_embedding(rel_path, emb_bytes)
            db.insert_audio_emotion(rel_path, emotion, json.dumps(probs))
            print(f"   ➕ Inserted audio: {filename}")
        except Exception as e:
            print(f"   ❌ Error audio: {e}")

    for filename in to_delete:
        rel_path = f"data/audio/{filename}"
        db.delete_audio(rel_path)
        print(f"   ➖ Removed missing audio: {filename}")


# ---------------------------------------------------
# 🚀 RUN ALL SYNC
# ---------------------------------------------------
def run_initial_sync():
    print("=======================================")
    print("🔄 Running initial filesystem sync...")
    print("=======================================")

    sync_images()
    sync_pdfs()
    sync_audio()

    print("✅ Initial sync COMPLETE!")

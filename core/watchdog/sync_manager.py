import os
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

from core.db.database_helper import DatabaseHelper
from core.image_search import ImageSearcher
from core.pdf_search import PDFSearcher
from core.emotion_model_v5 import EmotionModelV5


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "content_search_ai.db"
db = DatabaseHelper(str(DB_PATH))


# ======================================================
# 🟦 IMAGE SYNC
# ======================================================
def sync_images():
    watchdog = "images"
    db.update_watchdog_status(watchdog, "Running", "Starting image sync")

    try:
        searcher = ImageSearcher()
        model = searcher.image_model
        preprocess = searcher.preprocess
        device = searcher.device

        images_dir = BASE_DIR / "data/images"
        images_dir.mkdir(parents=True, exist_ok=True)

        fs_files = {
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        db_files = {
            Path(p).name for p in db.get_all_image_paths()
        }

        to_insert = fs_files - db_files
        to_delete = db_files - fs_files

        for filename in to_insert:
            full_path = images_dir / filename
            rel_path = f"data/images/{filename}"

            img = Image.open(full_path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            db.insert_image(
                filename,
                rel_path,
                emb.cpu().numpy().astype(np.float32).tobytes()
            )

            db.update_watchdog_status(
                watchdog,
                "Running",
                f"Inserted {filename}",
                inc_processed=True
            )

        for filename in to_delete:
            rel_path = f"data/images/{filename}"
            db.delete_image(rel_path)

            db.update_watchdog_status(
                watchdog,
                "Running",
                f"Removed {filename}",
                inc_processed=True
            )

        db.update_watchdog_status(watchdog, "Idle", "Image sync completed")

    except Exception as e:
        db.update_watchdog_status(watchdog, "Error", error=str(e))
        raise


# ======================================================
# 🟪 PDF SYNC
# ======================================================
def sync_pdfs():
    watchdog = "pdfs"
    db.update_watchdog_status(watchdog, "Running", "Starting PDF sync")

    try:
        pdf_dir = BASE_DIR / "data/pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        fs_files = {f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")}
        db_files = {Path(p).name for p in db.get_all_pdf_paths()}

        to_insert = fs_files - db_files
        to_delete = db_files - fs_files

        searcher = PDFSearcher(
            db_path=str(DB_PATH),
            model_path=str(BASE_DIR / "models/mclip_finetuned_coco_ready")
        )

        for filename in to_insert:
            full_path = pdf_dir / filename
            rel_path = f"data/pdfs/{filename}"

            pages = searcher.get_pdf_pages_embeddings(str(full_path))
            for page_no, emb, text in pages:
                db.insert_pdf_page(
                    rel_path,
                    page_no,
                    text,
                    emb.astype(np.float32).tobytes()
                )

            db.update_watchdog_status(
                watchdog,
                "Running",
                f"Inserted {filename}",
                inc_processed=True
            )

        for filename in to_delete:
            rel_path = f"data/pdfs/{filename}"
            db.delete_pdf(rel_path)

            db.update_watchdog_status(
                watchdog,
                "Running",
                f"Removed {filename}",
                inc_processed=True
            )

        db.update_watchdog_status(watchdog, "Idle", "PDF sync completed")

    except Exception as e:
        db.update_watchdog_status(watchdog, "Error", error=str(e))
        raise


# ======================================================
# 🟥 AUDIO SYNC
# ======================================================
def sync_audio():
    watchdog = "audio"
    db.update_watchdog_status(watchdog, "Running", "Starting audio sync")

    try:
        audio_dir = BASE_DIR / "data/audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        fs_files = {f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")}
        db_files = {Path(p).name for p in db.get_all_audio_paths()}

        to_insert = fs_files - db_files
        to_delete = db_files - fs_files

        whisper = WhisperModel("small", device="cpu", compute_type="int8")
        emotion_model = EmotionModelV5(
            ckpt_path=str(BASE_DIR / "models/best_model_audio_emotion_v5.pt")
        )
        mclip = SentenceTransformer(
            str(BASE_DIR / "models/mclip_finetuned_coco_ready"),
            device="cpu"
        )

        for filename in to_insert:
            full_path = audio_dir / filename
            rel_path = f"data/audio/{filename}"

            segments, _ = whisper.transcribe(str(full_path), beam_size=1)
            transcript = " ".join(seg.text for seg in segments).strip()

            emb = mclip.encode(transcript, normalize_embeddings=True)
            emotion, probs = emotion_model.predict(str(full_path))

            db.insert_audio_embedding(
                rel_path,
                emb.astype(np.float32).tobytes()
            )
            db.insert_audio_emotion(
                rel_path,
                emotion,
                json.dumps(probs)
            )

            db.update_watchdog_status(
                watchdog,
                "Running",
                f"Inserted {filename}",
                inc_processed=True
            )

        for filename in to_delete:
            rel_path = f"data/audio/{filename}"
            db.delete_audio(rel_path)

            db.update_watchdog_status(
                watchdog,
                "Running",
                f"Removed {filename}",
                inc_processed=True
            )

        db.update_watchdog_status(watchdog, "Idle", "Audio sync completed")

    except Exception as e:
        db.update_watchdog_status(watchdog, "Error", error=str(e))
        raise


# ======================================================
# 🚀 RUN ALL
# ======================================================
def run_initial_sync():
    print("=======================================")
    print("🔄 Running initial filesystem sync...")
    print("=======================================")

    sync_images()
    sync_pdfs()
    sync_audio()

    print("✅ Initial sync COMPLETE!")

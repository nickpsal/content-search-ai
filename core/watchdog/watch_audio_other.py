# core/watchdog/watch_audio_other.py
import time
import json
from pathlib import Path

import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import torch

from core.db.database_helper import DatabaseHelper
from core.emotion_model_v5 import EmotionModelV5


# ============================================
# 🔧 Convert absolute → relative DB path
# ============================================
def make_relative(full_path: str) -> str:
    full_path = full_path.replace("\\", "/")

    if "/data/" in full_path:
        rel = full_path.split("/data/")[1]
        return f"data/{rel}"

    print("❌ [AUDIO] Could not compute relative path for:", full_path)
    return None


# ============================================
# 🎧 AUDIO WATCHDOG HANDLER (audio_other)
# ============================================
class AudioOtherHandler(FileSystemEventHandler):
    def __init__(self):
        # root: content-search-ai
        self.base_dir = Path(__file__).resolve().parents[2]
        print("📌 AUDIO Watchdog BASE DIR =", self.base_dir)

        # -------------------------
        # DB
        # -------------------------
        db_path = self.base_dir / "content_search_ai.db"
        print("📌 AUDIO DB Path =", db_path)
        self.db = DatabaseHelper(str(db_path))

        #initialise Database
        self.db.initialise_database()

        # -------------------------
        # Paths
        # -------------------------
        self.audio_dir = self.base_dir / "data/audio/audio_other"
        self.watch_dir = str(self.audio_dir)

        # -------------------------
        # Models
        # -------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("🔹 Loading Faster-Whisper (CPU) for audio_other...")
        self.whisper = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("🔹 Loading EmotionModelV5 for audio_other...")
        self.emotion_model = EmotionModelV5(
            ckpt_path=str(self.base_dir / "models/best_model_audio_emotion_v5.pt"),
            device=self.device
        )

        print("🔹 Loading M-CLIP (SentenceTransformer) for audio embeddings...")
        self.mclip = SentenceTransformer(
            str(self.base_dir / "models/mclip_finetuned_coco_ready")
        )

        print("✅ AUDIO Watchdog models ready.\n")

    # --------------------------------------------
    # 🗑️ FILE DELETED
    # --------------------------------------------
    def on_deleted(self, event):
        if event.is_directory:
            return

        if not event.src_path.lower().endswith(".wav"):
            return

        rel_path = make_relative(event.src_path)
        if not rel_path:
            return

        print(f"🗑️ AUDIO Deleted → {rel_path}")

        # delete from both audio tables
        self.db.delete_audio(rel_path)

    # --------------------------------------------
    # 🆕 FILE CREATED
    # --------------------------------------------
    def on_created(self, event):
        if event.is_directory:
            return

        if not event.src_path.lower().endswith(".wav"):
            return

        full_path = event.src_path.replace("\\", "/")
        rel_path = make_relative(full_path)
        if not rel_path:
            return

        print(f"🆕 AUDIO Created → {rel_path}")

        try:
            # ----------------------------------------
            # 1️⃣ Transcription
            # ----------------------------------------
            segments, _ = self.whisper.transcribe(
                full_path,
                beam_size=1
            )
            transcript = " ".join(seg.text for seg in segments).strip()

            # ----------------------------------------
            # 2️⃣ Embedding with M-CLIP
            # ----------------------------------------
            emb = self.mclip.encode(
                transcript,
                normalize_embeddings=True
            ).astype(np.float32)
            emb_bytes = emb.tobytes()

            # ----------------------------------------
            # 3️⃣ Emotion Analysis
            # ----------------------------------------
            emotion, prob_dict = self.emotion_model.predict(full_path)
            emotion_json = json.dumps(prob_dict)

            filename = Path(full_path).name

            # ----------------------------------------
            # 4️⃣ Save to DB
            # ----------------------------------------
            self.db.insert_audio_embedding(rel_path, emb_bytes)
            self.db.insert_audio_emotion(rel_path, emotion, emotion_json)

            print(f"💾 AUDIO Saved to DB → {filename}")
            print(f"   • emotion: {emotion}")
            print(f"   • path   : {rel_path}")

        except Exception as e:
            print(f"❌ Error processing new audio {full_path}: {e}")


# ============================================
# 🚀 START AUDIO WATCHDOG
# ============================================
def start_watch():
    handler = AudioOtherHandler()
    observer = Observer()

    watch_dir = handler.watch_dir

    print("\n🎧 Watching AUDIO OTHER folder:")
    print(watch_dir)

    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

import os
import json
import sqlite3
import numpy as np
import soundfile as sf
import librosa
import sys

# Για να βλέπει το core/
sys.path.append(os.path.abspath("../.."))

from core.emotion_model_v5 import EmotionModelV5


DB_PATH = "../../content_search_ai.db"

# ----------- FIXED PATHS (ABSOLUTE) -----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIO_BASE = os.path.join(BASE_DIR, "data", "audio")

AUDIO_DIRS = [
    os.path.join(AUDIO_BASE, "AudioWAV"),
    os.path.join(AUDIO_BASE, "audio_other")
]

MODEL_CKPT = os.path.join(BASE_DIR, "models", "best_model_audio_emotion_v5.pt")
# ----------------------------------------------


def migrate_audio_emotions():
    print("🎧 Loading EmotionModelV5…")

    model = EmotionModelV5(
        ckpt_path=MODEL_CKPT,
        device="cpu"
    )

    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    cursor = conn.cursor()

    audio_files = []

    for d in AUDIO_DIRS:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".wav"):
                full_path = os.path.join(d, f)
                audio_files.append(full_path.replace("\\", "/"))  # clean slashes

    print(f"📥 Found {len(audio_files)} audio files.")

    imported = 0

    for audio_path in audio_files:
        print(f"🔍 Processing {audio_path}...")
        try:
            emotion, prob_dict = model.predict(audio_path)

            cursor.execute(
                """
                INSERT INTO audio_emotions (audio_path, emotion, emotion_scores)
                VALUES (?, ?, ?)
                """,
                (
                    audio_path,
                    emotion,
                    json.dumps(prob_dict)
                )
            )

            imported += 1

        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")

    conn.commit()
    conn.close()

    print("\n🎉 Emotion migration completed!")
    print(f"📌 Total emotions imported: {imported}")


if __name__ == "__main__":
    migrate_audio_emotions()

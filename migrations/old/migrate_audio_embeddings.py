import os
import sqlite3
import numpy as np

# ============================================
# Correct paths based on your project
# ============================================
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(BASE, "content_search_ai.db")
EMBEDS_DIR = os.path.join(BASE, "data/transcripts/embeds")

AUDIO_DIRS = [
    os.path.join(BASE, "data/audio/AudioWAV"),
    os.path.join(BASE, "data/audio/audio_other")
]


# ============================================
# Helper: find audio file across folders
# ============================================
def find_audio_file(filename):
    for folder in AUDIO_DIRS:
        full = os.path.join(folder, filename)
        if os.path.isfile(full):
            return full
    return None


# ============================================
# Insert embedding
# ============================================
def insert_embedding(cursor, audio_path, vector):
    blob = vector.astype(np.float32).tobytes()

    cursor.execute("""
        INSERT INTO audio_embeddings (audio_path, embedding)
        VALUES (?, ?)
    """, (audio_path, blob))


# ============================================
# Main migration
# ============================================
def migrate_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    files = [f for f in os.listdir(EMBEDS_DIR) if f.endswith(".npy")]

    print(f"📥 Found {len(files)} embedding files to import.")

    imported = 0

    for f in files:
        npy_path = os.path.join(EMBEDS_DIR, f)

        # Example: "1001_DFA_ANG_XX.wav.npy" → "1001_DFA_ANG_XX.wav"
        original_audio = f.replace(".npy", "")

        # Try locating the real audio file
        audio_path = find_audio_file(original_audio)

        if audio_path is None:
            print(f"⚠️ Audio missing → storing embedding anyway: {original_audio}")
            audio_path = f"missing:{original_audio}"

        # Load vector from .npy
        vector = np.load(npy_path)

        insert_embedding(cursor, audio_path, vector)
        imported += 1

    conn.commit()
    conn.close()

    print("\n🎉 Embedding migration completed!")
    print(f"📌 Total embeddings imported: {imported}")


# ============================================
# Run
# ============================================
if __name__ == "__main__":
    migrate_embeddings()

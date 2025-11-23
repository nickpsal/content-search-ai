import sqlite3
import os

DB_PATH = "../../content_search_ai.db"

def fix_transcript_paths():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audio_transcripts'")
    if cursor.fetchone() is None:
        print("❌ Table audio_transcripts does not exist!")
        return

    cursor.execute("SELECT id, audio_path FROM audio_transcripts")
    rows = cursor.fetchall()

    updated = 0

    for row_id, path in rows:
        if path is None:
            continue

        # Normalize Windows slashes
        p = path.replace("\\", "/")

        # Fix wrong prefix ./data/audio/main/
        if "/audio/main/" in p:
            filename = p.split("/main/")[1]  # get "1001_XXXX.wav"
            new_path = f"data/audio/AudioWAV/{filename}"
        else:
            new_path = p

        if new_path != path:
            cursor.execute(
                "UPDATE audio_transcripts SET audio_path = ? WHERE id = ?",
                (new_path, row_id)
            )
            updated += 1

    conn.commit()
    conn.close()

    print(f"🎉 Fixed transcript paths! Updated: {updated}")

if __name__ == "__main__":
    fix_transcript_paths()

import sqlite3
import csv
import os

DB_PATH = "../content_search_ai.db"

TRANSCRIPTS_FILES = [
    "../data/transcripts/transcripts_main.csv",
    "../data/transcripts/transcripts_other.csv"
]

AUDIO_BASE = "../data/audio"   # <-- Ρύθμισε το αν τα audio είναι αλλού

def insert_transcript(cursor, audio_path, transcript):
    cursor.execute(
        """
        INSERT INTO audio_transcripts (audio_path, transcript)
        VALUES (?, ?)
        """,
        (audio_path, transcript)
    )

def migrate_transcripts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total_imported = 0

    for csv_file in TRANSCRIPTS_FILES:
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            continue

        print(f"\n📥 Loading transcripts from: {csv_file}")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"   → Found {len(rows)} transcripts")

            for row in rows:
                filename = row.get("filename")
                folder = row.get("folder")
                transcript = row.get("transcript")

                if not filename or not folder or not transcript:
                    print(f"⚠️ Skipping invalid row: {row}")
                    continue

                # Build full audio path
                audio_path = os.path.join(AUDIO_BASE, folder, filename)

                insert_transcript(cursor, audio_path, transcript)
                total_imported += 1

        conn.commit()
        print(f"✅ Imported transcripts from: {csv_file}")

    conn.close()
    print(f"\n🎉 Migration completed successfully!")
    print(f"📌 Total transcripts imported: {total_imported}")

if __name__ == "__main__":
    migrate_transcripts()

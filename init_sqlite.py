import sqlite3
import os

DB_PATH = "content_search_ai.db"

SCHEMA = """
-- ============================
-- TABLE: images
-- ============================
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    image_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================
-- TABLE: pdf_pages
-- ============================
CREATE TABLE IF NOT EXISTS pdf_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_path TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text_content TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================
-- TABLE: audio_transcripts
-- ============================
CREATE TABLE IF NOT EXISTS audio_transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path TEXT NOT NULL,
    transcript TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================
-- TABLE: audio_emotions
-- ============================
CREATE TABLE IF NOT EXISTS audio_emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path TEXT NOT NULL,
    emotion TEXT NOT NULL,
    emotion_scores TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================
-- TABLE: search_logs
-- ============================
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_type TEXT NOT NULL,
    query_input TEXT NOT NULL,
    results TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def init_db():
    # Δημιουργία DB αν δεν υπάρχει
    if not os.path.exists(DB_PATH):
        print(f"📁 Creating new database at {DB_PATH}")
    else:
        print(f"📁 Database already exists at {DB_PATH}, ensuring tables exist...")

    # Σύνδεση / δημιουργία
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Εκτέλεση schema
    cursor.executescript(SCHEMA)

    # Αποθήκευση & κλείσιμο
    conn.commit()
    conn.close()

    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    init_db()

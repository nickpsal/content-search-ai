import sqlite3

DB_PATH = "../content_search_ai.db"

SQL = """
CREATE TABLE IF NOT EXISTS audio_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL)
    conn.commit()
    conn.close()
    print("✅ audio_embeddings table created successfully!")

if __name__ == "__main__":
    create_table()

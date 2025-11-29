import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "content_search_ai.db"


class DatabaseHelper:
    def __init__(self, db_path):
        self.db_path = db_path
        self.initialise_database()

    # =========================================
    #          INITIALISE DATABASE
    # =========================================
    def initialise_database(self):
        if os.path.exists(self.db_path):
            print("🗄️ Database already exists → OK")
            return

        print(f"🆕 Creating database at: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # ---------------------------
        # IMAGES
        # ---------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT UNIQUE,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ---------------------------
        # PDF PAGES
        # ---------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdf_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_path TEXT,
                page_number INTEGER,
                text_content TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ---------------------------
        # AUDIO EMBEDDINGS
        # ---------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT UNIQUE,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ---------------------------
        # AUDIO EMOTIONS
        # ---------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT UNIQUE,
                emotion TEXT,
                emotion_scores TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ---------------------------
        # SEARCH LOGS
        # ---------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                modality TEXT,
                results TEXT,
                searched_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        conn.close()
        print("✅ Database created successfully.")

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    # =========================================
    #                IMAGES
    # =========================================
    def insert_image(self, filename, filepath, embedding):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO images (filename, filepath, embedding)
            VALUES (?, ?, ?)
        """, (filename, filepath, embedding))
        conn.commit()
        conn.close()

    def delete_image(self, filepath):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM images WHERE filepath = ?", (filepath,))
        conn.commit()
        conn.close()

    def get_all_image_filenames(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT filename FROM images")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_all_image_paths(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT filepath FROM images")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    # =========================================
    #                  PDFs
    # =========================================
    def insert_pdf_page(self, pdf_path, page_number, text_content, embedding_bytes):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO pdf_pages (pdf_path, page_number, text_content, embedding)
            VALUES (?, ?, ?, ?)
        """, (pdf_path, page_number, text_content, embedding_bytes))
        conn.commit()
        conn.close()

    def delete_pdf(self, pdf_path):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM pdf_pages WHERE pdf_path = ?", (pdf_path,))
        conn.commit()
        conn.close()

    def get_all_pdf_paths(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT pdf_path FROM pdf_pages")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    # =========================================
    #                  AUDIO
    # =========================================
    def insert_audio_embedding(self, audio_path, embedding_bytes):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO audio_embeddings (audio_path, embedding)
            VALUES (?, ?)
        """, (audio_path, embedding_bytes))
        conn.commit()
        conn.close()

    def insert_audio_emotion(self, audio_path, emotion, emotion_scores_json):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO audio_emotions (audio_path, emotion, emotion_scores)
            VALUES (?, ?, ?)
        """, (audio_path, emotion, emotion_scores_json))
        conn.commit()
        conn.close()

    def delete_audio(self, audio_path):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM audio_embeddings WHERE audio_path = ?", (audio_path,))
        cur.execute("DELETE FROM audio_emotions WHERE audio_path = ?", (audio_path,))
        conn.commit()
        conn.close()

    def get_all_audio_paths(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT audio_path FROM audio_embeddings")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

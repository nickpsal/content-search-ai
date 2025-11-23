# core/db/database_helper.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "content_search_ai.db"


class DatabaseHelper:
    def __init__(self, db_path):
        self.db_path = db_path

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    # =========================================
    #                 IMAGES
    # =========================================

    def delete_image(self, rel_path):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM images WHERE image_path = ?", (rel_path,))
        conn.commit()
        conn.close()

    def insert_image(self, filename, rel_path, embedding):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO images (filename, image_path, embedding)
            VALUES (?, ?, ?)
        """, (filename, rel_path, embedding))
        conn.commit()
        conn.close()

    # =========================================
    #                   PDFs
    # =========================================

    def delete_pdf(self, rel_path):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM pdf_pages WHERE pdf_path = ?", (rel_path,))
        conn.commit()
        conn.close()

    def insert_pdf_page(self, rel_path, page_number, text, embedding_bytes):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO pdf_pages (pdf_path, page_number, text_content, embedding)
            VALUES (?, ?, ?, ?)
        """, (rel_path, page_number, text, embedding_bytes))
        conn.commit()
        conn.close()

    # =========================================
    #                  AUDIO
    # =========================================

    # ------- AUDIO EMBEDDINGS -------
    def insert_audio_embedding(self, audio_path, embedding_bytes):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audio_embeddings (audio_path, embedding)
            VALUES (?, ?)
        """, (audio_path, embedding_bytes))
        conn.commit()
        conn.close()

    # ------- AUDIO EMOTIONS -------
    def insert_audio_emotion(self, audio_path, emotion, emotion_scores_json):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audio_emotions (audio_path, emotion, emotion_scores)
            VALUES (?, ?, ?)
        """, (audio_path, emotion, emotion_scores_json))
        conn.commit()
        conn.close()

    # ------- DELETE AUDIO (ALL TABLES) -------
    def delete_audio(self, audio_path):
        conn = self._get_conn()
        cur = conn.cursor()

        # from embeddings
        cur.execute("DELETE FROM audio_embeddings WHERE audio_path = ?", (audio_path,))

        # from emotions
        cur.execute("DELETE FROM audio_emotions WHERE audio_path = ?", (audio_path,))

        conn.commit()
        conn.close()

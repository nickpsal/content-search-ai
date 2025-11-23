# core/db/database_helper.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "content_search_ai.db"

class DatabaseHelper:
    def __init__(self, db_path):
        self.db_path = db_path

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    # -------------------------
    # DELETE IMAGE
    # -------------------------
    def delete_image(self, rel_path):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM images WHERE image_path = ?", (rel_path,))
        conn.commit()
        conn.close()

    # -------------------------
    # INSERT IMAGE
    # -------------------------
    def insert_image(self, filename, rel_path, embedding):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO images (filename, image_path, embedding)
            VALUES (?, ?, ?)
        """, (filename, rel_path, embedding))
        conn.commit()
        conn.close()

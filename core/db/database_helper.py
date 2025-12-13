import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

# Default DB path
DB_PATH = Path(__file__).resolve().parents[2] / "content_search_ai.db"


class DatabaseHelper:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.initialise_database()

    # =========================================
    #              CONNECTION
    # =========================================
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # =========================================
    #          INITIALISE DATABASE
    # =========================================
    def initialise_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = self._get_conn()
        cur = conn.cursor()

        # IMAGES
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT UNIQUE,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # PDF PAGES
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

        # AUDIO
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT UNIQUE,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT UNIQUE,
                emotion TEXT,
                emotion_probs TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # SEARCH LOGS
        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                modality TEXT,
                results TEXT,
                searched_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # WATCHDOG STATUS
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watchdog_status (
                name TEXT PRIMARY KEY,
                status TEXT,
                last_event TEXT,
                last_updated REAL,
                processed_count INTEGER DEFAULT 0,
                error TEXT
            );
        """)

        for name in ("images", "pdfs", "audio"):
            cur.execute("""
                INSERT OR IGNORE INTO watchdog_status
                (name, status, last_event, last_updated, processed_count, error)
                VALUES (?, 'Idle', NULL, strftime('%s','now'), 0, NULL)
            """, (name,))

        conn.commit()
        conn.close()

    # =========================================
    #               COUNTS
    # =========================================
    def count_images(self):
        return self._count("images")

    def count_pdf_pages(self):
        return self._count("pdf_pages")

    def count_pdfs(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT pdf_path) AS c FROM pdf_pages")
        c = cur.fetchone()["c"]
        conn.close()
        return c

    def count_audio(self):
        return self._count("audio_embeddings")

    def _count(self, table):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) AS c FROM {table}")
        c = cur.fetchone()["c"]
        conn.close()
        return c

    # =========================================
    #               PATH LISTS
    # =========================================
    def get_all_image_paths(self):
        return self._get_column("images", "filepath")

    def get_all_pdf_paths(self):
        return self._get_column("pdf_pages", "pdf_path", distinct=True)

    def get_all_audio_paths(self):
        return self._get_column("audio_embeddings", "audio_path")

    def _get_column(self, table, col, distinct=False):
        conn = self._get_conn()
        cur = conn.cursor()
        q = f"SELECT {'DISTINCT' if distinct else ''} {col} FROM {table}"
        cur.execute(q)
        rows = cur.fetchall()
        conn.close()
        return [r[col] for r in rows]

    # =========================================
    #                  IMAGES
    # =========================================
    def insert_image(self, filename, filepath, embedding):
        self._insert(
            "images",
            ("filename", "filepath", "embedding"),
            (filename, filepath, embedding)
        )

    def delete_image(self, filepath):
        self._delete("images", "filepath", filepath)

    # =========================================
    #                  PDFs
    # =========================================
    def insert_pdf_page(self, pdf_path, page_number, text_content, embedding):
        self._insert(
            "pdf_pages",
            ("pdf_path", "page_number", "text_content", "embedding"),
            (pdf_path, page_number, text_content, embedding)
        )

    def delete_pdf(self, pdf_path):
        self._delete("pdf_pages", "pdf_path", pdf_path)

    # =========================================
    #                  AUDIO
    # =========================================
    def insert_audio_embedding(self, audio_path, embedding):
        self._insert(
            "audio_embeddings",
            ("audio_path", "embedding"),
            (audio_path, embedding)
        )

    def insert_audio_emotion(self, audio_path, emotion, emotion_probs):
        self._insert(
            "audio_emotions",
            ("audio_path", "emotion", "emotion_probs"),
            (audio_path, emotion, emotion_probs)
        )

    def delete_audio(self, audio_path):
        self._delete("audio_embeddings", "audio_path", audio_path)
        self._delete("audio_emotions", "audio_path", audio_path)

    # =========================================
    #            WATCHDOG STATUS
    # =========================================
    def update_watchdog_status(
        self,
        name,
        status,
        last_event=None,
        error=None,
        inc_processed=False
    ):
        conn = self._get_conn()
        cur = conn.cursor()

        inc = 1 if inc_processed else 0

        cur.execute("""
            INSERT INTO watchdog_status
            (name, status, last_event, last_updated, processed_count, error)
            VALUES (?, ?, ?, strftime('%s','now'), ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                status = excluded.status,
                last_event = excluded.last_event,
                last_updated = excluded.last_updated,
                processed_count = watchdog_status.processed_count + ?,
                error = excluded.error
        """, (name, status, last_event, inc, error, inc))

        conn.commit()
        conn.close()

    def get_watchdog_status(self, name):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM watchdog_status WHERE name = ?", (name,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_watchdog_statuses(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM watchdog_status ORDER BY name")
        rows = cur.fetchall()
        conn.close()
        return {r["name"]: dict(r) for r in rows}

    # =========================================
    #               HELPERS
    # =========================================
    def _insert(self, table, cols, values):
        conn = self._get_conn()
        cur = conn.cursor()
        placeholders = ",".join("?" * len(values))
        cur.execute(
            f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            values
        )
        conn.commit()
        conn.close()

    def _delete(self, table, col, value):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {table} WHERE {col} = ?", (value,))
        conn.commit()
        conn.close()

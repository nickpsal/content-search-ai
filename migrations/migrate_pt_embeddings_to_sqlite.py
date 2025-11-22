import os
import sqlite3
import torch
import numpy as np

# ===========================================
# CONFIG
# ===========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "content_search_ai.db")

EMBEDDING_FILES = [
    os.path.join(BASE_DIR, "data", "embeddings", "val2017_image_embeddings.pt"),
    os.path.join(BASE_DIR, "data", "embeddings", "other_image_embeddings.pt"),
]

VAL_DIR = os.path.join(BASE_DIR, "data", "images", "val2017")
OTHER_DIR = os.path.join(BASE_DIR, "data", "images", "other")


# ===========================================
# FIND ACTUAL IMAGE PATH
# ===========================================
def find_image_path(filename):
    """Return absolute path if image exists in val2017 or other."""
    val_path = os.path.join(VAL_DIR, filename)
    if os.path.exists(val_path):
        return val_path

    other_path = os.path.join(OTHER_DIR, filename)
    if os.path.exists(other_path):
        return other_path

    print(f"⚠️ Image not found: {filename}")
    return None


# ===========================================
# INSERT INTO DB
# ===========================================
def insert_image(cursor, filename, image_path, embedding_blob):
    cursor.execute(
        """
        INSERT INTO images (filename, image_path, embedding)
        VALUES (?, ?, ?)
        """,
        (filename, image_path, embedding_blob),
    )


# ===========================================
# MAIN MIGRATION FUNCTION
# ===========================================
def migrate_embeddings():
    print(f"📁 Base dir: {BASE_DIR}")
    print(f"🗂️ Database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for file_path in EMBEDDING_FILES:
        print(f"\n📥 Loading embeddings from: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        data = torch.load(file_path, map_location="cpu")
        print(f"   → Found {len(data)} embeddings")

        for img_filename, embed in data.items():
            # Find full absolute path
            img_path = find_image_path(img_filename)
            if img_path is None:
                continue

            # Normalize path → always forward slashes
            abs_path = os.path.abspath(img_path).replace("\\", "/")

            # Tensor → numpy → float32 bytes
            if isinstance(embed, torch.Tensor):
                embed = embed.numpy()

            embed_blob = embed.astype(np.float32).tobytes()

            insert_image(cursor, img_filename, abs_path, embed_blob)

        conn.commit()
        print(f"✅ Imported all embeddings from: {file_path}")

    conn.close()
    print("\n🎉 Migration completed successfully!")


# ===========================================
# RUN
# ===========================================
if __name__ == "__main__":
    migrate_embeddings()

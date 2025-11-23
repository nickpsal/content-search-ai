-- ===========================================
-- TABLE: audio_embeddings
-- Stores audio embedding vectors (Whisper CLIP)
-- ===========================================
CREATE TABLE IF NOT EXISTS audio_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

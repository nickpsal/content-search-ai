#!/usr/bin/env python3
"""
AudioSearcher v2 – MCLIP Semantic Search + Emotion
-------------------------------------------------
Features:
- Faster-Whisper (CPU) για transcription
- M-CLIP για semantic search στα transcripts
- Emotion Model V5 για emotion detection
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel as FasterWhisper
from sentence_transformers import SentenceTransformer

from .emotion_model_v5 import EmotionModelV5


# ============================
# SAFE CSV LOADER
# ============================
def _safe_load_csv(path: Path):
    if not path.exists():
        return None

    if path.stat().st_size == 0:
        print(f"[WARN] Empty transcript file -> {path}, rebuilding")
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] Transcript file empty -> {path}, rebuilding")
            return None
        return df
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}, rebuilding")
        return None


# ============================
# LANGUAGE DETECTION
# ============================
def detect_language(text: str):
    for ch in text:
        if 'α' <= ch <= 'ω' or 'Α' <= ch <= 'Ω':
            return "el"
    return "en"


def _normalize_emotion_query(emotion_query: str) -> str:
    q = emotion_query.lower().strip()

    mapping = {
        "happy": "happy", "joy": "happy", "laugh": "happy",
        "χαρούμενος": "happy", "χαρά": "happy",

        "sad": "sad", "upset": "sad",
        "λυπημένος": "sad", "στεναχωρημένος": "sad",

        "angry": "angry", "mad": "angry",
        "θυμωμένος": "angry", "θυμός": "angry",

        "fearful": "fearful", "scared": "fearful",
        "φοβισμένος": "fearful", "φόβος": "fearful",

        "disgust": "disgust", "disgusted": "disgust",
        "αηδιασμένος": "disgust", "αηδία": "disgust",

        "neutral": "neutral", "calm": "neutral",
        "ουδέτερος": "neutral",
    }

    return mapping.get(q, q)


# ============================
# AUDIO SEARCHER
# ============================
class AudioSearcher:
    def __init__(
        self,
        audio_main="data/audio/AudioWAV",
        audio_other="data/audio/audio_other",
        emotion_model_path="models/best_model_audio_emotion_v5.pt",
        transcripts_main="data/transcripts/transcripts_main.csv",
        transcripts_other="data/transcripts/transcripts_other.csv",
        mclip_model_path="models/mclip_finetuned_coco_ready",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_main = Path(audio_main)
        self.audio_other = Path(audio_other)
        self.emotion_model_path = Path(emotion_model_path)
        self.transcripts_main_path = Path(transcripts_main)
        self.transcripts_other_path = Path(transcripts_other)
        self.mclip_model_path = mclip_model_path

        self.emotion_cache_path = Path("data/emotions/emotion_cache.csv")
        self.emotion_cache = self._load_emotion_cache()

        self.transcripts_main_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcripts_other_path.parent.mkdir(parents=True, exist_ok=True)

        self.emb_dir = Path("data/transcripts/embeds")
        self.emb_dir.mkdir(parents=True, exist_ok=True)

        print("=======================================")
        print(f"[INFO] Device           : {self.device}")
        print(f"[INFO] Audio MAIN       : {self.audio_main}")
        print(f"[INFO] Audio OTHER      : {self.audio_other}")
        print(f"[INFO] Emotion model    : {self.emotion_model_path}")
        print(f"[INFO] M-CLIP model     : {self.mclip_model_path}")
        print("=======================================\n")

        self._load_models()

        self.transcripts = None
        self.transcripts_main = None
        self.transcripts_other = None

    # ============================
    # LOAD MODELS
    # ============================
    def _load_models(self):
        print("[INFO] Loading Faster-Whisper (CPU)")
        self.fw = FasterWhisper(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("[INFO] Loading Emotion Model V5")
        self.emotion_model = EmotionModelV5(
            self.emotion_model_path,
            device=self.device
        )

        print("[INFO] Loading M-CLIP")

        self.mclip = SentenceTransformer(
            self.mclip_model_path,
            device=self.device,
            model_kwargs={
                "device_map": None,
                "low_cpu_mem_usage": False,
                "torch_dtype": torch.float32
            }
        )

        print("[INFO] Models loaded\n")

    # ============================
    # EMOTION CACHE
    # ============================
    def _load_emotion_cache(self):
        if not self.emotion_cache_path.exists():
            return {}

        df = pd.read_csv(self.emotion_cache_path)
        cache = {}

        for _, row in df.iterrows():
            cache[row["filename"]] = {
                "emotion": row["emotion"],
                "emotion_probs": {
                    "angry": row["angry"],
                    "disgust": row["disgust"],
                    "fearful": row["fearful"],
                    "happy": row["happy"],
                    "neutral": row["neutral"],
                    "sad": row["sad"],
                }
            }
        return cache

    def save_emotion_cache(self):
        self.emotion_cache_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for fname, data in self.emotion_cache.items():
            probs = data["emotion_probs"]
            rows.append({
                "filename": fname,
                "emotion": data["emotion"],
                **probs
            })

        pd.DataFrame(rows).to_csv(self.emotion_cache_path, index=False)

    # ============================
    # TRANSCRIBE
    # ============================
    def transcribe_audio(self, path: Path) -> str:
        segs, _ = self.fw.transcribe(str(path), beam_size=1)
        return " ".join([s.text for s in segs]).strip()

    # ============================
    # BUILD TRANSCRIPTS + EMBEDS
    # ============================
    def build_all_transcripts(self):
        df_other = _safe_load_csv(self.transcripts_other_path)
        if df_other is None:
            rows = []
            for w in tqdm(self.audio_other.glob("*.wav"), desc="Transcribing OTHER"):
                rows.append({
                    "filename": w.name,
                    "folder": "other",
                    "transcript": self.transcribe_audio(w)
                })
            df_other = pd.DataFrame(rows)
            df_other.to_csv(self.transcripts_other_path, index=False)

        df_main = _safe_load_csv(self.transcripts_main_path)
        if df_main is None:
            rows = []
            for w in tqdm(self.audio_main.glob("*.wav"), desc="Transcribing MAIN"):
                rows.append({
                    "filename": w.name,
                    "folder": "main",
                    "transcript": self.transcribe_audio(w)
                })
            df_main = pd.DataFrame(rows)
            df_main.to_csv(self.transcripts_main_path, index=False)

        self.transcripts = pd.concat([df_other, df_main], ignore_index=True)

        print("[INFO] Building MCLIP embeddings")
        for _, row in tqdm(self.transcripts.iterrows(), total=len(self.transcripts)):
            emb_path = self.emb_dir / f"{row['filename']}.npy"
            if not emb_path.exists():
                emb = self.mclip.encode(row["transcript"], normalize_embeddings=True)
                np.save(emb_path, emb)

        print("[INFO] Transcripts and embeddings ready\n")
        return self.transcripts

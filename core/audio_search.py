#!/usr/bin/env python3
"""
AudioSearcher v2 – MCLIP Semantic Search + Emotion
---------------------------------------------------
Features:
- Faster-Whisper (CPU) για transcription (μόνο για build/update transcripts)
- M-CLIP (mclip_finetuned_coco_ready) για semantic search πάνω στα transcripts
- Emotion Model V5 (best_model_audio_emotion_v5.pt) για emotion detection
- Υποστηρίζει:
    - build/update transcripts (main + other)
    - build MCLIP embeddings για όλα τα transcripts
    - semantic search σε EL/EN (και γενικά multilingual)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel as FasterWhisper
from sentence_transformers import SentenceTransformer

# ============================
#    EMOTION MODEL IMPORT
# ============================
from .emotion_model_v5 import EmotionModelV5


# ============================
#   SAFE CSV LOADER
# ============================
def _safe_load_csv(path: Path):
    """Load a CSV safely – return None if empty or invalid."""
    if not path.exists():
        return None

    if path.stat().st_size == 0:
        print(f"⚠️ Empty transcript file → {path}, rebuilding...")
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ Transcript file empty → {path}, rebuilding...")
            return None
        return df
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}, rebuilding...")
        return None


# -------------------------------------------------
# Detect query language (EL / EN)
# -------------------------------------------------
def detect_language(text: str):
    for ch in text:
        if 'α' <= ch <= 'ω' or 'Α' <= ch <= 'Ω':
            return "el"
    return "en"

def _normalize_emotion_query(emotion_query: str) -> str:
    """
    Μετατρέπει το input του χρήστη σε ένα από τα επίσημα labels:
    angry, disgust, fearful, happy, neutral, sad
    Υποστηρίζει και αγγλικά και ελληνικά και synonyms.
    """
    q = emotion_query.lower().strip()

    # Λεξικό mapping (ENG + GR synonyms)
    mapping = {
        # HAPPY
        "happy": "happy",
        "joy": "happy",
        "laugh": "happy",
        "laughed": "happy",
        "laughing": "happy",
        "joyful": "happy",
        "χαρούμενος": "happy",
        "χαρά": "happy",

        # SAD
        "sad": "sad",
        "upset": "sad",
        "crying": "sad",
        "cry": "sad",
        "λυπημένος": "sad",
        "στεναχωρημένος": "sad",

        # ANGRY
        "angry": "angry",
        "mad": "angry",
        "furious": "angry",
        "rage": "angry",
        "θυμωμένος": "angry",
        "θυμός": "angry",

        # FEARFUL
        "fearful": "fearful",
        "scared": "fearful",
        "afraid": "fearful",
        "φοβισμένος": "fearful",
        "φόβος": "fearful",

        # DISGUST
        "disgust": "disgust",
        "disgusted": "disgust",
        "αηδιασμένος": "disgust",
        "αηδία": "disgust",

        # NEUTRAL
        "neutral": "neutral",
        "calm": "neutral",
        "flat": "neutral",
        "ουδέτερος": "neutral",
    }

    return mapping.get(q, q)

# ============================
#       AUDIO SEARCHER
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
        """
        AudioSearcher v2
        - audio_main: βασικό dataset (π.χ. RAVDESS/CREMA-D)
        - audio_other: custom χρήστη (ηχογραφήσεις κλπ)
        - emotion_model_path: path στο best_model_audio_emotion_v5.pt
        - mclip_model_path: path στο M-CLIP fine-tuned model (SentenceTransformer)
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        self.query = ""
        self.audio_main = Path(audio_main)
        self.audio_other = Path(audio_other)
        self.emotion_model_path = Path(emotion_model_path)
        self.transcripts_main_path = Path(transcripts_main)
        self.transcripts_other_path = Path(transcripts_other)
        self.mclip_model_path = mclip_model_path
        self.emotion_cache_path = Path("data/emotions/emotion_cache.csv")
        self.emotion_cache = self._load_emotion_cache()

        # Ensure dirs
        self.transcripts_main_path.parent.mkdir(exist_ok=True, parents=True)
        self.transcripts_other_path.parent.mkdir(exist_ok=True, parents=True)

        # Embeddings directory
        self.emb_dir = Path("data/transcripts/embeds")
        self.emb_dir.mkdir(parents=True, exist_ok=True)

        print("=======================================")
        print(f"🚀 Device            : {self.device}")
        print(f"📂 Audio MAIN        : {self.audio_main}")
        print(f"📂 Audio OTHER       : {self.audio_other}")
        print(f"🧠 Emotion model     : {self.emotion_model_path}")
        print(f"🔤 M-CLIP model path : {self.mclip_model_path}")
        print("=======================================\n")

        self._load_models()

        # Will be loaded lazy
        self.transcripts = None
        self.transcripts_main = None
        self.transcripts_other = None

    # ==========================================
    #  LOAD MODELS
    # ==========================================
    def _load_models(self):
        print("🔹 Loading Faster-Whisper (CPU transcription)...")
        self.fw = FasterWhisper(
            "small",
            device="cpu",           # ALWAYS CPU (σταθερό + χωρίς cuDNN θέματα)
            compute_type="int8"     # Fast & lightweight
        )

        print("🔹 Loading Emotion Model V5...")
        self.emotion_model = EmotionModelV5(
            self.emotion_model_path,
            device=self.device
        )

        print("🔹 Loading M-CLIP (SentenceTransformer)...")
        self.mclip = SentenceTransformer(self.mclip_model_path)
        print("✅ All models loaded.\n")

    # ==========================================
    #  EMOTION CACHE FILE
    # ==========================================
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
                "angry": probs["angry"],
                "disgust": probs["disgust"],
                "fearful": probs["fearful"],
                "happy": probs["happy"],
                "neutral": probs["neutral"],
                "sad": probs["sad"],
            })

        pd.DataFrame(rows).to_csv(self.emotion_cache_path, index=False)

    # ==========================================
    #  TRANSCRIBE SINGLE FILE
    # ==========================================
    def transcribe_audio(self, path: Path) -> str:
        segs, _ = self.fw.transcribe(str(path), beam_size=1)
        return " ".join([s.text for s in segs]).strip()

    # ==========================================
    #  BUILD TRANSCRIPTS + MCLIP EMBEDDINGS
    # ==========================================
    def build_all_transcripts(self):
        """
        - Φτιάχνει / φορτώνει transcripts για:
            - audio_other → transcripts_other.csv
            - audio_main  → transcripts_main.csv
        - Κρατάει self.transcripts ως merged DataFrame
        - Φτιάχνει MCLIP embeddings (.npy) για ΟΛΑ τα transcripts
        """

        # ---------- OTHER ----------
        df_other = _safe_load_csv(self.transcripts_other_path)
        if df_other is None:
            rows = []
            wavs = sorted(self.audio_other.glob("*.wav"))
            if not wavs:
                print(f"ℹ️ No .wav files found in OTHER folder: {self.audio_other}")
            for w in tqdm(wavs, desc="Transcribing OTHER"):
                rows.append({
                    "filename": w.name,
                    "folder": "other",
                    "transcript": self.transcribe_audio(w)
                })
            df_other = pd.DataFrame(rows)
            df_other.to_csv(self.transcripts_other_path, index=False)
        self.transcripts_other = df_other

        # ---------- MAIN ----------
        df_main = _safe_load_csv(self.transcripts_main_path)
        if df_main is None:
            rows = []
            wavs = sorted(self.audio_main.glob("*.wav"))
            if not wavs:
                print(f"ℹ️ No .wav files found in MAIN folder: {self.audio_main}")
            for w in tqdm(wavs, desc="Transcribing MAIN"):
                rows.append({
                    "filename": w.name,
                    "folder": "main",
                    "transcript": self.transcribe_audio(w)
                })
            df_main = pd.DataFrame(rows)
            df_main.to_csv(self.transcripts_main_path, index=False)
        self.transcripts_main = df_main

        # ---------- MERGE ----------
        self.transcripts = pd.concat([df_other, df_main], ignore_index=True)

        # ---------- BUILD MCLIP EMBEDDINGS ----------
        print("\n🔹 Building MCLIP embeddings for all transcripts...\n")
        for _, row in tqdm(
            self.transcripts.iterrows(),
            total=self.transcripts.shape[0],
            desc="MCLIP embeddings"
        ):
            fname = row["filename"]
            emb_path = self.emb_dir / f"{fname}.npy"

            # Αν υπάρχει ήδη, το αφήνουμε (γρήγορο rebuild)
            if emb_path.exists():
                continue

            text = row["transcript"]
            emb = self.mclip.encode(text, normalize_embeddings=True)
            np.save(emb_path, emb)

        print("✅ Transcripts + embeddings ready.\n")
        return self.transcripts

    # ==========================================
    #  MCLIP SEMANTIC SEARCH + EMOTION
    # ==========================================
    def search_semantic_emotion(self, query: str, top_k=10):
        """
        MCLIP Semantic Search + Emotion + Language Penalty
        ---------------------------------------------------
        - embed query with MCLIP
        - compute cosine similarity with transcript embeddings
        - apply language mismatch penalty
        - return top_k most relevant results
        - run Emotion Model V5 ONLY on top_k
        """

        # Ensure transcripts & embeddings exist
        if self.transcripts is None:
            self.build_all_transcripts()

        print(f"\n🔍 MCLIP Semantic Search for: '{query}'\n")

        query_lang = detect_language(query)
        print(f"🌐 Query language detected: {query_lang.upper()}")

        # -------------------------------------------------
        # Encode query once
        # -------------------------------------------------
        q_emb = self.mclip.encode(query, normalize_embeddings=True)

        results = []

        # -------------------------------------------------
        # Compute similarity with every transcript embedding
        # -------------------------------------------------
        for _, row in tqdm(
                self.transcripts.iterrows(),
                total=self.transcripts.shape[0],
                desc="Computing similarities"
        ):
            fname = row["filename"]
            emb_path = self.emb_dir / f"{fname}.npy"

            if not emb_path.exists():
                continue

            t_emb = np.load(emb_path)

            # Base similarity (cosine)
            sim = float(np.dot(q_emb, t_emb))

            # Detect transcript language
            text_lang = detect_language(row["transcript"])

            # -------------------------------------------------
            # 🚨 LANGUAGE PENALTY
            # -------------------------------------------------
            if query_lang != text_lang:
                sim -= 1.0  # Strong language penalty

            folder = row["folder"]
            full_path = (
                self.audio_other / fname if folder == "other"
                else self.audio_main / fname
            )

            results.append({
                "filename": fname,
                "folder": folder,
                "full_path": str(full_path),
                "transcript": row["transcript"],
                "similarity": sim,
                "text_language": text_lang
            })

        # -------------------------------------------------
        # Sort by similarity
        # -------------------------------------------------
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        # 🔥 Remove negative scores
        results = [r for r in results if r["similarity"] > 0]

        top_results = results[:top_k]

        # -------------------------------------------------
        # Run Emotion Model only on top_k results
        # -------------------------------------------------
        print("\n🔹 Running Emotion Model on top results...\n")

        for r in top_results:
            emo_label, emo_probs = self.emotion_model.predict(Path(r["full_path"]))
            r["emotion"] = emo_label
            r["emotion_probs"] = emo_probs

        print("\n✅ Semantic search complete.\n")
        return top_results

    def search_by_emotion(self, emotion_query: str, top_k=10):
        # transcripts must exist
        if self.transcripts is None:
            self.build_all_transcripts()

        target = _normalize_emotion_query(emotion_query)

        results = []
        updated_cache = False

        for _, row in self.transcripts.iterrows():
            fname = row["filename"]

            # ALWAYS define full_path first
            full_path = (
                self.audio_other / fname
                if row["folder"] == "other"
                else self.audio_main / fname
            )

            # CACHE HIT
            if fname in self.emotion_cache:
                emo_label = self.emotion_cache[fname]["emotion"]
                emo_probs = self.emotion_cache[fname]["emotion_probs"]

            else:
                # CACHE MISS → run model ONCE
                emo_label, emo_probs = self.emotion_model.predict(full_path)

                self.emotion_cache[fname] = {
                    "emotion": emo_label,
                    "emotion_probs": emo_probs
                }
                updated_cache = True

            score = emo_probs.get(target, 0)

            results.append({
                "filename": fname,
                "folder": row["folder"],
                "full_path": str(full_path),  # ← always exists now
                "transcript": row["transcript"],
                "emotion": emo_label,
                "emotion_probs": emo_probs,
                "similarity": score
            })

        if updated_cache:
            self.save_emotion_cache()

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_query_segments(self, audio_path: Path, query: str):
        q = query.lower().strip()
        if not q:
            return []

        segments, _ = self.fw.transcribe(
            str(audio_path),
            beam_size=1,
            word_timestamps=True
        )

        hits = []

        for seg in segments:
            if not seg.words:
                continue

            for w in seg.words:
                if q in w.word.lower():
                    hits.append((float(w.start), float(w.end)))

        return hits

    # ============================
    #       QUERY TYPE
    # ============================
    def classify_query_type(self, query:str):
        self.query = query
        q = self.query.lower().strip()

        # Emotion keywords & synonyms
        emotion_keywords = {
            "angry", "anger", "mad", "furious", "rage",
            "sad", "sadness", "crying", "upset",
            "happy", "joy", "laugh", "laughing", "smile",
            "fearful", "scared", "afraid", "fear",
            "disgust", "disgusted", "gross",
            "neutral", "calm", "flat"
        }

        # Greek emotions
        greek_emotions = {
            "θυμωμένος", "θυμός",
            "λυπημένος", "στεναχωρημένος",
            "χαρούμενος", "χαρά",
            "φοβισμένος", "φόβος",
            "αηδιασμένος", "αηδία",
            "ουδέτερος"
        }

        if any(word in q for word in emotion_keywords.union(greek_emotions)):
            return "emotion"

        return "keyword"

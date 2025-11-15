#!/usr/bin/env python3
"""
AudioSearcher (Faster-Whisper version)
-------------------------------------
- Whisper encoder for audio embeddings (your fine-tuned model)
- Faster-Whisper for transcription
- M-CLIP for text embeddings
- Unified search API
"""

import torch
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel as FasterWhisper
from transformers import WhisperProcessor, WhisperModel
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F


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


class AudioSearcher:
    def __init__(
        self,
        audio_main="data/AudioWAV",
        audio_other="data/audio_other",
        model_path="models/best_model_audio_emotion_v4.pt",
        mclip_path="models/mclip_finetuned_coco_ready",
        emb_main="data/embeddings/audio_embeddings_main.pt",
        emb_other="data/embeddings/audio_embeddings_other.pt",
        transcripts_main="data/transcripts/transcripts_main.csv",
        transcripts_other="data/transcripts/transcripts_other.csv",
        device=None
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        self.audio_main = Path(audio_main)
        self.audio_other = Path(audio_other)
        self.model_path = Path(model_path)
        self.mclip_path = Path(mclip_path)
        self.emb_main_path = Path(emb_main)
        self.emb_other_path = Path(emb_other)
        self.transcripts_main_path = Path(transcripts_main)
        self.transcripts_other_path = Path(transcripts_other)

        # Ensure folders
        self.emb_main_path.parent.mkdir(exist_ok=True)
        self.emb_other_path.parent.mkdir(exist_ok=True)
        self.transcripts_main_path.parent.mkdir(exist_ok=True)
        self.transcripts_other_path.parent.mkdir(exist_ok=True)

        print(f"🚀 Device: {self.device}")
        print(f"📂 Audio main:  {self.audio_main}")
        print(f"📂 Audio other: {self.audio_other}")
        print(f"📦 Model:        {self.model_path}")

        self._load_models()

        self.emb_main = None
        self.emb_other = None
        self.files_main = None
        self.files_other = None

        self.transcripts_main = None
        self.transcripts_other = None
        self.transcripts = None


    # ---------------------------------------------------------
    # Load models: Whisper encoder + Faster-Whisper + MCLIP
    # ---------------------------------------------------------
    def _load_models(self):
        print("🔹 Loading Whisper encoder...")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-small").to(self.device)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper.eval()

        print("🔹 Loading Faster-Whisper (transcription)...")
        self.fw = FasterWhisper(
            "small",
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
        )

        print("🔹 Loading M-CLIP...")
        self.text_model = SentenceTransformer(str(self.mclip_path)).to(self.device)
        self.text_model.eval()

        print("🔹 Loading projection layers...")
        ckpt = torch.load(self.model_path, map_location=self.device)

        # Audio projection
        self.audio_proj = nn.Linear(self.whisper.config.d_model, 512).to(self.device)
        self.audio_proj.load_state_dict(ckpt["audio_proj"])
        self.audio_proj.eval()

        # Text projection
        self.text_proj = nn.Linear(self.text_model.get_sentence_embedding_dimension(), 512).to(self.device)
        self.text_proj.load_state_dict(ckpt["text_proj"])
        self.text_proj.eval()


    # ---------------------------------------------------------
    # Load + preprocess audio
    # ---------------------------------------------------------
    @torch.no_grad()
    def load_audio(self, path: Path):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        return inputs


    # ---------------------------------------------------------
    # Audio Embeddings
    # ---------------------------------------------------------
    @torch.no_grad()
    def audio_embedding(self, path: Path):
        try:
            inputs = self.load_audio(path)
            feats = self.whisper.encoder(inputs).last_hidden_state.mean(dim=1)
            proj = self.audio_proj(feats)
            return F.normalize(proj, p=2, dim=-1).cpu()
        except Exception as e:
            print(f"⚠️ Error embedding {path.name}: {e}")
            return None


    # ---------------------------------------------------------
    # Text Embeddings
    # ---------------------------------------------------------
    @torch.no_grad()
    def text_embedding(self, text: str):
        emb = self.text_model.encode([text], convert_to_tensor=True, device=self.device)
        proj = self.text_proj(emb)
        return F.normalize(proj, p=2, dim=-1).cpu()


    # ---------------------------------------------------------
    # Embeddings builder
    # ---------------------------------------------------------
    def _build_embeddings_for_folder(self, folder: Path, out: Path):
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            print(f"⚠️ No WAV in {folder}")
            return None, None

        embs, files = [], []
        print(f"🎧 Found {len(wavs)} files in {folder}")

        for w in tqdm(wavs, desc=f"Embedding {folder.name}"):
            e = self.audio_embedding(w)
            if e is not None:
                embs.append(e)
                files.append(w.name)

        if not embs:
            return None, None

        full = torch.cat(embs, dim=0)
        torch.save({"embeddings": full, "files": files}, out)

        print(f"💾 Saved → {out}")
        return full, files

    def build_all_embeddings(self):
        # MAIN embeddings
        if not self.emb_main_path.exists():
            self.emb_main, self.files_main = self._build_embeddings_for_folder(
                self.audio_main, self.emb_main_path
            )
        else:
            print("🔹 MAIN audio embeddings already exist — skipping.")

        # OTHER embeddings
        if not self.emb_other_path.exists():
            self.emb_other, self.files_other = self._build_embeddings_for_folder(
                self.audio_other, self.emb_other_path
            )
        else:
            print("🔹 OTHER audio embeddings already exist — skipping.")

    def load_embeddings(self):
        if self.emb_main_path.exists():
            d = torch.load(self.emb_main_path)
            self.emb_main = d["embeddings"]
            self.files_main = d["files"]

        if self.emb_other_path.exists():
            d = torch.load(self.emb_other_path)
            self.emb_other = d["embeddings"]
            self.files_other = d["files"]

        print("📥 Loaded embeddings:",
              f"\n  MAIN:  {len(self.files_main) if self.files_main else 0}",
              f"\n  OTHER: {len(self.files_other) if self.files_other else 0}")


    # ---------------------------------------------------------
    # Faster-Whisper transcription
    # ---------------------------------------------------------
    @torch.no_grad()
    def transcribe_audio(self, path: Path):
        segments, info = self.fw.transcribe(str(path), beam_size=1)
        text = " ".join([seg.text for seg in segments])
        return text.strip()


    # ---------------------------------------------------------
    # Build transcripts
    # ---------------------------------------------------------
    def build_all_transcripts(self):
        # ---------- MAIN ----------
        df_main = _safe_load_csv(self.transcripts_main_path)
        if df_main is None:
            rows = []
            for w in tqdm(sorted(self.audio_main.glob("*.wav")), desc="Transcribing MAIN"):
                rows.append({
                    "filename": w.name,
                    "folder": "main",
                    "transcript": self.transcribe_audio(w)
                })
            df_main = pd.DataFrame(rows)
            df_main.to_csv(self.transcripts_main_path, index=False, encoding="utf-8")

        self.transcripts_main = df_main

        # ---------- OTHER ----------
        df_other = _safe_load_csv(self.transcripts_other_path)
        if df_other is None:
            rows = []
            for w in tqdm(sorted(self.audio_other.glob("*.wav")), desc="Transcribing OTHER"):
                rows.append({
                    "filename": w.name,
                    "folder": "other",
                    "transcript": self.transcribe_audio(w)
                })
            df_other = pd.DataFrame(rows)
            df_other.to_csv(self.transcripts_other_path, index=False, encoding="utf-8")

        self.transcripts_other = df_other

        # MERGE
        self.transcripts = pd.concat([df_main, df_other], ignore_index=True)
        return self.transcripts

    # ---------------------------------------------------------
    # Vector search
    # ---------------------------------------------------------
    def search_audio(self, text: str, top_k=10):
        if self.emb_main is None or self.emb_other is None:
            self.load_embeddings()

        all_emb = torch.cat([self.emb_main, self.emb_other], dim=0)
        all_files = self.files_main + self.files_other

        q = self.text_embedding(text)
        sims = F.cosine_similarity(q, all_emb).numpy()

        idxs = sims.argsort()[::-1][:top_k]
        return [(all_files[i], float(sims[i])) for i in idxs]


    # ---------------------------------------------------------
    # Transcript search
    # ---------------------------------------------------------
    def search_transcripts(self, query: str, top_k=10):
        if self.transcripts is None:
            self.build_all_transcripts()

        df = self.transcripts.copy()
        df["score"] = df["transcript"].str.lower().apply(lambda t: t.count(query.lower()))
        df = df[df["score"] > 0].sort_values("score", ascending=False)

        return df.head(top_k)[["filename", "transcript", "score"]].to_dict("records")

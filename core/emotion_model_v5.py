import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import whisper

# Canonical emotion classes
EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]


# ============================================================
# 🔊 Audio Emotion Model (Whisper encoder + classifier)
# ============================================================
class AudioEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Whisper encoder — ΠΑΝΤΑ CPU (σταθερό & ασφαλές)
        self.whisper_model = whisper.load_model("small", device="cpu")

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # Emotion classifier
        self.classifier = nn.Linear(256, len(EMOTIONS))

    def forward(self, mel):
        # Whisper encoder forward (χωρίς gradients)
        with torch.no_grad():
            enc = self.whisper_model.encoder(mel)

        # Mean pooling over time
        z = enc.mean(dim=1)

        # Projection + classification
        z = self.proj(z)
        return self.classifier(z)


# ============================================================
# 🎭 EmotionModelV5 — Inference Wrapper
# ============================================================
class EmotionModelV5:
    def __init__(self, ckpt_path: str):
        # Emotion model ΠΑΝΤΑ CPU
        self.device = "cpu"

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Build model
        self.model = AudioEmotionModel()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path: str):
        # ----------------------------------------
        # 1️⃣ Load audio
        # ----------------------------------------
        audio, sr = sf.read(audio_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)

        # ----------------------------------------
        # 2️⃣ Whisper preprocessing
        # ----------------------------------------
        audio = whisper.pad_or_trim(
            torch.tensor(audio, dtype=torch.float32)
        )

        mel = whisper.log_mel_spectrogram(audio)
        mel = mel.unsqueeze(0)  # (1, 80, T)
        mel = mel.to(self.device)

        # ----------------------------------------
        # 3️⃣ Forward pass
        # ----------------------------------------
        logits = self.model(mel)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # ----------------------------------------
        # 4️⃣ Output
        # ----------------------------------------
        idx = int(np.argmax(probs))

        return EMOTIONS[idx], {
            EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))
        }

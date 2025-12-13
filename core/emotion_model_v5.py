import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import whisper

EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]


class AudioEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔴 Whisper ΠΑΝΤΑ CPU, εκτός torch graph
        self.whisper_model = whisper.load_model("small", device="cpu")

        self.proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.classifier = nn.Linear(256, len(EMOTIONS))

    def forward(self, mel):
        with torch.no_grad():
            enc = self.whisper_model.encoder(mel)

        z = enc.mean(dim=1)
        z = self.proj(z)
        return self.classifier(z)


class EmotionModelV5:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = device

        print(f"[INFO] Loading Emotion Model V5: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model = AudioEmotionModel()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path):
        audio, sr = sf.read(audio_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)

        audio = whisper.pad_or_trim(
            torch.tensor(audio, dtype=torch.float32)
        )

        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(self.device)

        logits = self.model(mel)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        return EMOTIONS[idx], {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

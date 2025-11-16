import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf


EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]


class AudioEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        import whisper
        self.whisper_model = whisper.load_model("small")

        # Whisper-small encoder dim = 768
        self.proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.classifier = nn.Linear(256, len(EMOTIONS))

    def forward(self, mel):
        enc = self.whisper_model.encoder(mel)
        z = enc.mean(dim=1)
        z_proj = self.proj(z)
        logits = self.classifier(z_proj)
        return logits


class EmotionModelV5:
    def __init__(self, ckpt_path, device="cpu"):
        import whisper
        self.device = device

        print(f"Loading Emotion Model V5 from: {ckpt_path}")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)

        # Create the model and load weights
        self.model = AudioEmotionModel().to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.whisper = whisper   # used to access pad_or_trim + mel

    @torch.no_grad()
    def predict(self, audio_path):
        # Load + resample
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Whisper preprocessing
        audio = self.whisper.pad_or_trim(torch.tensor(audio, dtype=torch.float32))
        mel = self.whisper.log_mel_spectrogram(audio).unsqueeze(0).to(self.device)

        # Model forward
        logits = self.model(mel)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        emotion = EMOTIONS[idx]

        prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

        return emotion, prob_dict

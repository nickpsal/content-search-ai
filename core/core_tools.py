import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pathlib import Path

# ============================
# BASE / DATA / DB PATHS
# ============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "content_search_ai.db"


class CoreTools:
    def __init__(self, audio_path: str):
        """
        audio_path = absolute path (resolved via resolve_data_path)
        """
        self.audio_path = audio_path

        self.EMOTION_COLORS = {
            "angry": "#FF4B4B",
            "disgust": "#8E44AD",
            "fearful": "#9B59B6",
            "happy": "#2ECC71",
            "neutral": "#95A5A6",
            "sad": "#3498DB",
        }

    # ==========================================================
    # WAVEFORM + SPECTROGRAM
    # ==========================================================
    def plot_waveform_and_spectrogram_with_highlights(
        self,
        query_segments=None,
        emotion_label=None
    ):
        """
        Left → waveform with emotion color background + query highlights
        Right → mel spectrogram
        """

        if query_segments is None:
            query_segments = []

        # Load audio
        y, sr = librosa.load(self.audio_path, sr=None)
        duration = len(y) / sr

        # Emotion background color
        emo_color = self.EMOTION_COLORS.get(emotion_label, "#F9E79F")

        # Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        # -------------------------
        # LEFT: WAVEFORM
        # -------------------------
        ax[0].axvspan(0, duration, color=emo_color, alpha=0.18)
        librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#1f77b4")

        ax[0].set_title(f"Waveform (Emotion: {emotion_label})")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")

        # Highlights for transcript query
        for (start, end) in query_segments:
            start = max(0.0, start)
            end = min(duration, end)
            if end > start:
                ax[0].axvspan(start, end, color="orange", alpha=0.4)

        # -------------------------
        # RIGHT: SPECTROGRAM
        # -------------------------
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax[1], cmap="magma"
        )
        ax[1].set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax[1], format="%+2.f dB")

        st.pyplot(fig)

    # ==========================================================
    # Alias method for clarity
    # ==========================================================
    def plot_waveform_and_spectrogram_with_emotion_highlight(
        self,
        query_segments=None,
        emotion_label=None
    ):
        return self.plot_waveform_and_spectrogram_with_highlights(
            query_segments=query_segments,
            emotion_label=emotion_label
        )

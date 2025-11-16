import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

class CoreTools:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.EMOTION_COLORS = {
            "angry": "#FF4B4B",
            "disgust": "#8E44AD",
            "fearful": "#9B59B6",
            "happy": "#2ECC71",
            "neutral": "#95A5A6",
            "sad": "#3498DB",
        }

    def plot_waveform_and_spectrogram_with_highlights(self,  query_segments=None, emotion_label=None):
        """
        Σχεδιάζει:
          - Αριστερά: waveform
          - Δεξιά: mel spectrogram
        και αν δοθούν highlight_segments = [(start, end), ...],
        βάφει τα αντίστοιχα intervals πάνω στο waveform.
        """
        if query_segments is None:
            query_segments = []

            # Load audio
        y, sr = librosa.load(self.audio_path, sr=None)
        duration = len(y) / sr

        # Emotion color
        emo_color = self.EMOTION_COLORS.get(emotion_label, "#F9E79F")  # default pale yellow

        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        # ---------------------------------------
        # LEFT: WAVEFORM + EMOTION BACKGROUND
        # ---------------------------------------
        ax[0].axvspan(0, duration, color=emo_color, alpha=0.18)  # emotion color full area

        librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#1f77b4")
        ax[0].set_title(f"Waveform  (Emotion: {emotion_label})")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")

        # QUERY HIGHLIGHT (orange)
        for (start, end) in query_segments:
            s = max(0.0, start)
            e = min(duration, end)
            if e > s:
                ax[0].axvspan(s, e, color="orange", alpha=0.4)

        # ---------------------------------------
        # RIGHT: MEL SPECTROGRAM (normal)
        # ---------------------------------------
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax[1], cmap="magma"
        )
        ax[1].set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax[1], format="%+2.f dB")

        st.pyplot(fig)

    def plot_waveform_and_spectrogram_with_emotion_highlight(
            self,
            query_segments=None,
            emotion_label=None
    ):
        """
        - Waveform (left)
        - Mel Spectrogram (right)
        - Highlight query segments
        - Color waveform background based on emotion
        """
        if query_segments is None:
            query_segments = []

        # Load audio
        y, sr = librosa.load(self.audio_path, sr=None)
        duration = len(y) / sr

        # Emotion color
        emo_color = EMOTION_COLORS.get(emotion_label, "#F9E79F")  # default pale yellow

        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        # ---------------------------------------
        # LEFT: WAVEFORM + EMOTION BACKGROUND
        # ---------------------------------------
        ax[0].axvspan(0, duration, color=emo_color, alpha=0.18)  # emotion color full area

        librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#1f77b4")
        ax[0].set_title(f"Waveform  (Emotion: {emotion_label})")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")

        # QUERY HIGHLIGHT (orange)
        for (start, end) in query_segments:
            s = max(0.0, start)
            e = min(duration, end)
            if e > s:
                ax[0].axvspan(s, e, color="orange", alpha=0.4)

        # ---------------------------------------
        # RIGHT: MEL SPECTROGRAM (normal)
        # ---------------------------------------
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax[1], cmap="magma"
        )
        ax[1].set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax[1], format="%+2.f dB")

        st.pyplot(fig)

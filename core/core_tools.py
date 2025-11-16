import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

class CoreTools:
    def __init__(self, audio_path):
        self.audio_path = audio_path

    def plot_waveform_and_spectrogram(self):
        # Load audio
        y, sr = librosa.load(self.audio_path, sr=None)

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Create side-by-side layout
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        # === LEFT: WAVEFORM ===
        librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#1f77b4")
        ax[0].set_title("Waveform")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")

        # === RIGHT: MEL SPECTROGRAM ===
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax[1], cmap="magma")
        ax[1].set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax[1], format="%+2.f dB")

        # Render in Streamlit
        st.pyplot(fig)


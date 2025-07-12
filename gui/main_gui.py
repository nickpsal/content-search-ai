# gui/main_gui.py
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import ttk

# Δημιουργία βασικού παραθύρου με θέμα
app = tb.Window(themename="darkly")  # "darkly", "cosmo", "journal", "solar", κ.ά.
app.title("Content Search AI")
app.geometry("900x600")

# Δημιουργία των tabs (Notebook)
notebook = ttk.Notebook(app)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# ---------- Tabs ----------
# 1. Settings Tab
settings_tab = ttk.Frame(notebook)
notebook.add(settings_tab, text="Settings")

# 2. Image Tab
image_tab = ttk.Frame(notebook)
notebook.add(image_tab, text="Image")

# 3. Audio Tab
audio_tab = ttk.Frame(notebook)
notebook.add(audio_tab, text="Audio")

# ---------- Περιεχόμενο (μόνο για δείγμα) ----------
tb.Label(settings_tab, text="Ρυθμίσεις συστήματος", font=("Segoe UI", 12)).pack(pady=10)
tb.Label(image_tab, text="Αναζήτηση εικόνας", font=("Segoe UI", 12)).pack(pady=10)
tb.Label(audio_tab, text="Αναζήτηση ήχου", font=("Segoe UI", 12)).pack(pady=10)

# Εκκίνηση του GUI
app.mainloop()

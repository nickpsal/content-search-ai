# gui/main_gui.py
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import ttk


# Εισαγωγή tabs
from settings_tab import create_settings_tab
# from tabs.image_tab import create_image_tab
# from tabs.audio_tab import create_audio_tab

# Δημιουργία βασικού παραθύρου με θέμα
app = tb.Window(themename="darkly")  # "darkly", "cosmo", "journal", "solar", κ.ά.
app.title("Content Search AI")
app.geometry("900x600")

# Δημιουργία των tabs (Notebook)
notebook = ttk.Notebook(app)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# ---------- Tabs ----------

# Δημιουργία κάθε Tab
# create_image_tab(notebook)
# create_audio_tab(notebook)

# 1. Settings Tab
create_settings_tab(notebook)


# 2. Image Tab
image_tab = ttk.Frame(notebook)
notebook.add(image_tab, text="Image")

# 3. Audio Tab
audio_tab = ttk.Frame(notebook)
notebook.add(audio_tab, text="Audio")

# Εκκίνηση του GUI
app.mainloop()

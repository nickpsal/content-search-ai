# gui/main_gui.py
import ttkbootstrap as tb
from tkinter import ttk

# Εισαγωγή tabs
from settings_tab import create_settings_tab
from image_tab import create_image_tab
# from audio_tab import create_audio_tab

def center_window(window, width=900, height=900):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    window.geometry(f"{width}x{height}+{x}+{y}")

# Δημιουργία βασικού παραθύρου με θέμα
app = tb.Window(themename="darkly")  # "darkly", "cosmo", "journal", "solar", κ.ά.
app.title("Content Search AI")
center_window(app, 900, 600)

# Δημιουργία των tabs (Notebook)
notebook = ttk.Notebook(app)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# ------------------------------------- Tabs --------------------------------------#
create_image_tab(notebook)
create_settings_tab(notebook)
# create_audio_tab(notebook)

# Εκκίνηση του GUI

app.mainloop()

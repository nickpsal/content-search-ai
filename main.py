import os
from concurrent.futures import ThreadPoolExecutor
from core.watchdog.watch_images_other import start_watch
from core.watchdog.watch_pdfs import start_watch
from core.watchdog.watch_audio_other import start_watch
from core.watchdog.sync_manager import run_initial_sync

# ============================
# WATCHDOG – IMAGES
# ============================
def run_watchdog_images():
    print("🚀 Watchdog Images started!")
    start_watch()

# ============================
# WATCHDOG – PDFs
# ============================
def run_watchdog_pdfs():
    print("📄 Watchdog PDFs started!")
    start_watch()

# ============================
# WATCHDOG – AUDIO (other)
# ============================
def run_watchdog_audio():
    print("🎧 Watchdog AUDIO started!")
    start_watch()

# ============================
# STREAMLIT
# ============================
def run_streamlit():
    print("🚀 Streamlit started!")
    os.system("streamlit run app.py --server.port=8501 --server.address=0.0.0.0")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    # 4 workers για 4 ανεξάρτητες εργασίες
    with ThreadPoolExecutor(max_workers=5) as executor:
        run_initial_sync()
        executor.submit(run_watchdog_images)
        executor.submit(run_watchdog_pdfs)
        executor.submit(run_watchdog_audio)
        executor.submit(run_streamlit)

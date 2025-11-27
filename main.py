import os
from concurrent.futures import ThreadPoolExecutor

# ============================
# WATCHDOG – IMAGES
# ============================
def run_watchdog_images():
    print("🚀 Watchdog Images started!")
    from core.watchdog.watch_images_other import start_watch
    start_watch()

# ============================
# WATCHDOG – PDFs
# ============================
def run_watchdog_pdfs():
    print("📄 Watchdog PDFs started!")
    from core.watchdog.watch_pdfs import start_watch
    start_watch()

# ============================
# WATCHDOG – AUDIO (other)
# ============================
def run_watchdog_audio():
    print("🎧 Watchdog AUDIO started!")
    from core.watchdog.watch_audio_other import start_watch
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
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(run_watchdog_images)
        executor.submit(run_watchdog_pdfs)
        executor.submit(run_watchdog_audio)
        executor.submit(run_streamlit)

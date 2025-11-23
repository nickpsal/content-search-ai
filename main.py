import multiprocessing
import os


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

    # Windows = must use spawn
    multiprocessing.set_start_method("spawn")

    # -----------------------------------
    # PROCESS 1 → Images Watchdog
    # -----------------------------------
    p_images = multiprocessing.Process(
        target=run_watchdog_images,
        daemon=False
    )

    # -----------------------------------
    # PROCESS 2 → PDFs Watchdog
    # -----------------------------------
    p_pdfs = multiprocessing.Process(
        target=run_watchdog_pdfs,
        daemon=False
    )

    # -----------------------------------
    # PROCESS 3 → AUDIO Watchdog
    # -----------------------------------
    p_audio = multiprocessing.Process(
        target=run_watchdog_audio,
        daemon=False
    )

    # -----------------------------------
    # PROCESS 4 → Streamlit
    # -----------------------------------
    p_streamlit = multiprocessing.Process(
        target=run_streamlit,
        daemon=False
    )

    # Start all
    p_images.start()
    p_pdfs.start()
    p_audio.start()
    p_streamlit.start()

    # Wait for all to finish
    p_images.join()
    p_pdfs.join()
    p_audio.join()
    p_streamlit.join()

import subprocess
import multiprocessing
import os


def run_watchdog_images():
    print("🚀 Watchdog Images started!")
    from core.watchdog.watch_images_other import start_watch
    start_watch()


def run_watchdog_pdfs():
    print("📄 Watchdog PDFs started!")
    from core.watchdog.watch_pdfs import start_watch
    start_watch()


def run_streamlit():
    print("🚀 Streamlit started!")
    os.system("streamlit run app.py --server.port=8501 --server.address=0.0.0.0")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # -------------------------
    # PROCESS 1 → IMAGES
    # -------------------------
    p_images = multiprocessing.Process(
        target=run_watchdog_images,
        daemon=False
    )

    # -------------------------
    # PROCESS 2 → PDFs
    # -------------------------
    p_pdfs = multiprocessing.Process(
        target=run_watchdog_pdfs,
        daemon=False
    )

    # -------------------------
    # PROCESS 3 → Audios
    # -------------------------

    # -------------------------
    # PROCESS 3 → STREAMLIT
    # -------------------------
    p_streamlit = multiprocessing.Process(
        target=run_streamlit,
        daemon=False
    )

    # Start
    p_images.start()
    p_pdfs.start()
    p_streamlit.start()

    # Join
    p_images.join()
    p_pdfs.join()
    p_streamlit.join()

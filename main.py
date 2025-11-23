import threading
import subprocess


def run_watchdog():
    from core.watchdog.watch_images_other import start_watch
    print("🚀 Watchdog started!")
    start_watch()


def run_streamlit():
    print("🚀 Streamlit started!")
    subprocess.call([
        "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])


if __name__ == "__main__":

    t1 = threading.Thread(target=run_watchdog, daemon=True)
    t2 = threading.Thread(target=run_streamlit, daemon=False)

    t1.start()
    t2.start()

    t2.join()     # κρατάει το streamlit ανοιχτό

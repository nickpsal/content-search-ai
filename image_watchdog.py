from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

IMAGE_DIR = "data/images"
PDF_DIR = "data/pdfs"
AUDIO_DIR = "data/audio"

class MultiHandler(FileSystemEventHandler):

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"🟢 [CREATED] {event.src_path}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"🔴 [DELETED] {event.src_path}")

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"🟡 [MODIFIED] {event.src_path}")

def start_watchdog():
    observer = Observer()
    handler = MultiHandler()

    observer.schedule(handler, IMAGE_DIR, recursive=False)
    # observer.schedule(handler, PDF_DIR, recursive=False)
    # observer.schedule(handler, AUDIO_DIR, recursive=False)

    observer.start()
    print("📡 Watchdog running on:")
    print(f"   → {IMAGE_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    start_watchdog()

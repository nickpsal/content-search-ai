import sys
import re

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.last_line_id = None

    def write(self, message):
        message = message.replace("\r", "")  # αφαίρεση carriage return
        lines = message.splitlines()

        for line in lines:
            if line.strip():  # αγνόησε κενές γραμμές
                self.text_widget.insert("end", line + "\n")
                self.text_widget.see("end")

    def flush(self):
        pass

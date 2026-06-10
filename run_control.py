"""Điều khiển dừng hợp tác: luồng nền nghe lệnh `quit`."""

import threading


class RunControl:
    def __init__(self, input_func=input):
        self.input_func = input_func
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._listen,
            name="research-run-control",
            daemon=True,
        )
        self._thread.start()

    def _listen(self):
        while not self._stop_event.is_set():
            try:
                command = self.input_func().strip().lower()
            except (EOFError, KeyboardInterrupt):
                command = "quit"
            if command == "quit":
                self._stop_event.set()

    def request_stop(self):
        self._stop_event.set()

    def stop_requested(self):
        return self._stop_event.is_set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

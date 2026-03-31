"""Progress reporting for SSE streaming."""

import queue
import threading


class ProgressReporter:
    """Thread-safe progress reporter.

    The pipeline thread calls emit() / emit_complete() / emit_error().
    The SSE endpoint reads dicts from the queue.
    """

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()

    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()

    def cancel(self):
        self.cancelled.set()

    def emit(self, percent: int, message: str):
        """Push a progress event."""
        self.queue.put({
            "percent": percent,
            "message": message,
        })

    def emit_complete(self, result: dict):
        """Push the final result event."""
        self.queue.put({
            "event": "complete",
            "percent": 100,
            "message": "Complete",
            "result": result,
        })

    def emit_error(self, message: str):
        """Push an error event."""
        self.queue.put({
            "event": "error",
            "message": message,
        })

    def sentinel(self):
        """Place a None sentinel to signal end-of-stream."""
        self.queue.put(None)

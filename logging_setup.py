"""Logging cho lượt nghiên cứu: console, rotating file và sự kiện trong DB."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from research_store import redact


class RunLogger:
    def __init__(self, run_id, logger, research_store, handlers):
        self.run_id = run_id
        self.logger = logger
        self.research_store = research_store
        self._handlers = handlers

    def event(self, event_type, message, payload=None, level=logging.INFO):
        safe_payload = redact(payload or {})
        rendered = f"[{event_type}] {message}"
        if safe_payload:
            rendered += " | " + json.dumps(
                safe_payload, ensure_ascii=False, sort_keys=True
            )
        self.logger.log(level, rendered)
        self.research_store.record_event(
            self.run_id, event_type, message, safe_payload
        )

    def close(self):
        for handler in self._handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


def create_run_logger(run_id, logs_dir, research_store, stream=None,
                      max_bytes=5_000_000, backup_count=5):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"research.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = RotatingFileHandler(
        logs_dir / f"{run_id}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)

    handlers = [file_handler, console_handler]
    for handler in handlers:
        logger.addHandler(handler)

    return RunLogger(run_id, logger, research_store, handlers)

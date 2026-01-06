import os
from typing import Any, Dict, Optional

from .utils import ensure_dir, jsonl_append


class JsonlLogger:
    def __init__(self, path: str) -> None:
        parent = os.path.dirname(path) or "."
        ensure_dir(parent)
        self.path = path

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec = {"event": event}
        if payload:
            rec.update(payload)
        jsonl_append(self.path, rec)

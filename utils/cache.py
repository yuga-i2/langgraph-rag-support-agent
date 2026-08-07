"""
Smart cache (stand-out feature).

Caches the *final structured response* for a normalised question so an
identical repeat question skips triage/retrieval/generation/verification
entirely. This is a pragmatic win for a support bot, where the same few
questions ("sync is broken", "can a viewer create a token") get asked
repeatedly.

Deliberately simple: a JSON file on disk, keyed by a hash of the
lower-cased/stripped question. No eviction policy is needed for an
assignment-scale deployment, but the hook is isolated here so swapping in
Redis/SQLite later is a one-file change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class QueryCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    @staticmethod
    def _key(question: str) -> str:
        normalised = " ".join(question.strip().lower().split())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        return self._data.get(self._key(question))

    def set(self, question: str, response: Dict[str, Any]) -> None:
        self._data[self._key(question)] = response
        try:
            self.path.write_text(json.dumps(self._data, indent=2))
        except OSError:
            pass  # cache is a best-effort optimisation, never fatal

    def clear(self) -> None:
        self._data = {}
        if self.path.exists():
            self.path.unlink()

"""Local JSON-based caching implementation with TTL and file locking."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Iterator, Optional

if sys.platform == "win32":
    import msvcrt

    @contextlib.contextmanager
    def _file_lock(file_handle: Any) -> Iterator[None]:
        """Acquire non-blocking lock on Windows; fall back gracefully if unsupported."""
        fd = file_handle.fileno()
        try:
            file_handle.seek(0)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                file_handle.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

else:
    import fcntl

    @contextlib.contextmanager
    def _file_lock(file_handle: Any) -> Iterator[None]:
        """Acquire exclusive lock on POSIX."""
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


class FileCache:
    """Thread-safe disk cache storing individual JSON files with expiration timestamps."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        default_ttl_seconds: int = 3600,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Initialize the file cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = ttl_seconds if ttl_seconds is not None else default_ttl_seconds
        self._thread_lock = threading.RLock()

    @staticmethod
    def hash_key(key: str) -> str:
        """Generate a SHA-256 hash string for an arbitrary key/query/URL."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _get_entry_path(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        hashed = self.hash_key(key)
        return self.cache_dir / f"{hashed}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve cached dictionary value if it exists and has not expired."""
        entry_path = self._get_entry_path(key)
        if not entry_path.exists():
            return None

        with self._thread_lock:
            if not entry_path.exists():
                return None

            try:
                with open(entry_path, "r+", encoding="utf-8") as f:
                    with _file_lock(f):
                        data = json.load(f)

                expires_at = data.get("expires_at", 0.0)
                if time.time() >= expires_at:
                    self._safe_delete(entry_path)
                    return None

                payload: Any = data.get("value")
                if isinstance(payload, dict):
                    return payload
                return {"value": payload}
            except (json.JSONDecodeError, OSError):
                return None

    def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_override: Optional[int] = None,
    ) -> None:
        """Persist a key-value pair to disk with an expiration timestamp."""
        ttl = ttl_override if ttl_override is not None else self.default_ttl
        now = time.time()
        record = {
            "key": key,
            "value": value,
            "created_at": now,
            "ttl": ttl,
            "expires_at": now + ttl,
        }

        entry_path = self._get_entry_path(key)

        with self._thread_lock:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.cache_dir,
                prefix="tmp_cache_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as tf:
                    with _file_lock(tf):
                        json.dump(record, tf, ensure_ascii=False)
                os.replace(temp_path, entry_path)
            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise

    def delete(self, key: str) -> bool:
        """Delete a single cached entry."""
        with self._thread_lock:
            entry_path = self._get_entry_path(key)
            return self._safe_delete(entry_path)

    def _safe_delete(self, path: Path) -> bool:
        """Remove file quietly if present."""
        try:
            if path.exists():
                path.unlink()
                return True
        except OSError:
            pass
        return False

    def clear_expired(self) -> int:
        """Scan cache directory and delete all expired entries."""
        removed_count = 0
        now = time.time()

        with self._thread_lock:
            for entry_path in self.cache_dir.glob("*.json"):
                try:
                    with open(entry_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    expires_at = data.get("expires_at", 0.0)
                    if now >= expires_at:
                        if self._safe_delete(entry_path):
                            removed_count += 1
                except (json.JSONDecodeError, OSError):
                    if self._safe_delete(entry_path):
                        removed_count += 1

        return removed_count

    def clear_all(self) -> int:
        """Delete all entries currently in the cache."""
        removed = 0
        with self._thread_lock:
            for entry_path in self.cache_dir.glob("*.json"):
                if self._safe_delete(entry_path):
                    removed += 1
        return removed

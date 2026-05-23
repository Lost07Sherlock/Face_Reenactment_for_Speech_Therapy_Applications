"""
file_manager.py
───────────────
Central registry for temporary files created during a session.

• Tracks every temp file written so they can be bulk-deleted on exit.
• Registers an atexit handler that fires when the Python process ends
  (covers Ctrl-C, normal exit, and Gradio/Streamlit server shutdown).
• Provides per-session subdirectories so parallel users don't collide.
• Thread-safe: uses a reentrant lock.
"""

import os
import glob
import atexit
import shutil
import threading
import time
import uuid

# ── Config ───────────────────────────────────────────────────────────────────
BASE_TEMP_DIRS = ["temp_uploads", "temp_patients", "animations"]

# ── Internal state ────────────────────────────────────────────────────────────
_lock             = threading.RLock()
_registered_files: set[str] = set()
_registered_dirs:  set[str] = set()


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _safe_remove(path: str) -> None:
    """Delete a single file silently."""
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _safe_rmtree(path: str) -> None:
    """Delete a directory tree silently."""
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all base temp directories (idempotent)."""
    for d in BASE_TEMP_DIRS:
        os.makedirs(d, exist_ok=True)


def make_session_dir(base: str = "temp_uploads") -> str:
    """
    Create and register a unique session subdirectory.

    Returns the full path so callers can write session-scoped files into it,
    e.g. ``temp_uploads/sess_3f2a…/source.jpg``.
    """
    sess_id = uuid.uuid4().hex[:12]
    path    = os.path.join(base, f"sess_{sess_id}")
    os.makedirs(path, exist_ok=True)
    register_temp_dir(path)
    return path


def register_temp_file(path: str) -> str:
    """Track *path* for cleanup at exit. Returns *path* unchanged."""
    with _lock:
        _registered_files.add(os.path.abspath(path))
    return path


def register_temp_dir(path: str) -> str:
    """Track *path* (directory) for cleanup at exit. Returns *path* unchanged."""
    with _lock:
        _registered_dirs.add(os.path.abspath(path))
    return path


def cleanup_file(path: str) -> None:
    """Immediately delete *path* and deregister it."""
    abs_path = os.path.abspath(path) if path else None
    _safe_remove(abs_path)
    with _lock:
        _registered_files.discard(abs_path)


def cleanup_all_temp() -> None:
    """
    Delete every tracked file and directory.
    Called automatically at process exit.
    """
    with _lock:
        files = set(_registered_files)
        dirs  = set(_registered_dirs)

    for f in files:
        _safe_remove(f)

    for d in dirs:
        _safe_rmtree(d)

    with _lock:
        _registered_files.clear()
        _registered_dirs.clear()

    # Also sweep base dirs of any stale leftovers
    for base in ["temp_uploads", "temp_patients"]:
        if not os.path.isdir(base):
            continue
        for item in os.listdir(base):
            full = os.path.join(base, item)
            if os.path.isfile(full):
                _safe_remove(full)
            elif os.path.isdir(full) and item.startswith("sess_"):
                _safe_rmtree(full)


def cleanup_old_sessions(max_age_hours: float = 2.0) -> int:
    """
    Remove session subdirectories older than *max_age_hours*.
    Returns the number of directories removed.
    Call this on app startup to recover disk space from crashed sessions.
    """
    removed = 0
    cutoff  = time.time() - max_age_hours * 3600

    for base in ["temp_uploads", "temp_patients"]:
        if not os.path.isdir(base):
            continue
        for entry in os.scandir(base):
            if entry.is_dir() and entry.name.startswith("sess_"):
                if entry.stat().st_mtime < cutoff:
                    _safe_rmtree(entry.path)
                    removed += 1

    return removed


def save_upload(file_obj, save_dir: str = "temp_uploads", prefix: str = "file") -> str | None:
    """
    Persist an uploaded file object to *save_dir* and register it for cleanup.

    Accepts either:
    • A path string (Gradio sometimes resolves uploads to a temp path).
    • Any object with a ``.name`` attribute (Streamlit ``UploadedFile``).
    • Any object with a ``.read()`` method.

    Returns the absolute path of the saved file, or ``None`` on failure.
    """
    if file_obj is None:
        return None

    ensure_dirs()
    os.makedirs(save_dir, exist_ok=True)

    # Gradio passes a filepath string directly
    if isinstance(file_obj, str) and os.path.isfile(file_obj):
        return register_temp_file(file_obj)

    # Derive a filename
    name = getattr(file_obj, "name", None) or f"{prefix}_{int(time.time())}.tmp"
    dest = os.path.join(save_dir, os.path.basename(name))

    # Write bytes
    try:
        if hasattr(file_obj, "getbuffer"):
            with open(dest, "wb") as f:
                f.write(file_obj.getbuffer())
        elif hasattr(file_obj, "read"):
            with open(dest, "wb") as f:
                f.write(file_obj.read())
        else:
            return None
    except (OSError, IOError):
        return None

    return register_temp_file(dest)


# ── Auto-cleanup registration ─────────────────────────────────────────────────
atexit.register(cleanup_all_temp)

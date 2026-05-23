"""
video_utils.py
──────────────
Shared video helper functions used by both the Gradio and Streamlit UIs.

Depends only on stdlib + OpenCV + FFmpeg (CLI) so it is UI-framework-agnostic.
"""

import os
import subprocess
import tempfile
import time
import cv2
from PIL import Image


# ── Duration & metadata ───────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """Return duration in seconds, or 0 on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps


def get_video_dims(video_path: str) -> tuple[int, int]:
    """Return (width, height), or (0, 0) on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frame(video_path: str, second: float = 0.0) -> Image.Image | None:
    """
    Extract a single frame at *second* seconds and return as a PIL Image (RGB).
    Falls back to the first available frame if the seek fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


# ── FFmpeg operations ─────────────────────────────────────────────────────────

def _run_ffmpeg(cmd: list[str]) -> bool:
    """Run an FFmpeg command. Returns True on success."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def trim_video(video_path: str, start: float, end: float) -> str:
    """
    Trim *video_path* to [start, end] seconds using FFmpeg.
    Returns the output path (``<stem>_trimmed.<ext>``).
    Passes *video_path* through unchanged if start == 0 and end == duration.
    """
    duration = get_video_duration(video_path)
    if start == 0 and (end == 0 or end >= duration):
        return video_path

    base, ext  = os.path.splitext(video_path)
    out_path   = f"{base}_trimmed_{int(start)}_{int(end)}{ext}"
    clip_dur   = end - start

    _run_ffmpeg([
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t",  str(clip_dur),
        "-c:v", "libx264", "-c:a", "aac",
        out_path,
    ])
    return out_path if os.path.isfile(out_path) else video_path


def crop_video(video_path: str, x: int, y: int, w: int, h: int) -> str:
    """
    Spatially crop *video_path* to the rectangle (x, y, w, h) using FFmpeg.
    Returns the output path (``<stem>_cropped.<ext>``).
    """
    base, ext = os.path.splitext(video_path)
    out_path  = f"{base}_cropped{ext}"

    _run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-vf",    f"crop={w}:{h}:{x}:{y}",
        "-c:a",   "copy",
        out_path,
    ])
    return out_path if os.path.isfile(out_path) else video_path


def extract_audio(video_path: str, out_path: str) -> bool:
    """
    Extract audio track from *video_path* into a 44.1 kHz stereo WAV.
    Returns True on success, False if no audio track exists or FFmpeg fails.
    """
    ok = _run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        out_path,
    ])
    return ok and os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def merge_audio_video(video_path: str, audio_path: str, out_path: str) -> bool:
    """
    Replace the audio track of *video_path* with *audio_path*.
    Writes result to *out_path*. Returns True on success.
    """
    return _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        out_path,
    ])


def re_encode_for_browser(video_path: str) -> str:
    """
    Re-encode *video_path* to H.264/AAC in an MP4 container so every browser
    can play it inline.  Returns the new path, or *video_path* on failure.
    """
    base, _  = os.path.splitext(video_path)
    out_path = f"{base}_web.mp4"
    ok = _run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path if ok and os.path.isfile(out_path) else video_path

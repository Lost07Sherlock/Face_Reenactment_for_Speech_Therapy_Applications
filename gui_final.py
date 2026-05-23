"""
gui_final.py  (combined Streamlit version)
──────────────────────────────────────────
Speech Therapy – LivePortrait Studio

This file merges the two previous versions so that BOTH dialogue-accuracy
methods are computed and contribute to the final weighted score:

  • Dialogue Accuracy (Acoustic)   — ContentVec embeddings + silero-vad +
                                      FastDTW (cosine).  Measures phonetic /
                                      acoustic similarity of the two clips.
  • Dialogue Accuracy (Transcript) — Whisper (run in its own conda env via
                                      subprocess) + word-error-rate.  Measures
                                      how close the spoken words are to the
                                      reference dialogue.

Final score = weighted sum of:
  • Articulation                 (default 30 %)
  • Dialogue Accuracy (Acoustic) (default 30 %)
  • Dialogue Accuracy (Transcript) (default 40 %)

Weights gracefully fall back if one of the two dialogue methods cannot run
(see _compute_weighted_final_score below).

All previous fixes preserved:
✓ Image crop no longer resets on every re-render
✓ Webcam recording is a proper state machine via callbacks
✓ Zero-second video bug fixed
✓ Face-alignment guide is always shown in the webcam preview
✓ Automatic temp-file cleanup via _SessionFiles
✓ Microphone audio captured alongside webcam video
✓ Audio/video sync via dynamic atempo adjustment
✓ Pre-evaluation guardrails (stationary, silent, duration sanity)
✓ Word-level diff display for transcript accuracy
✓ Whisper transcription language selector (Auto / English / Hindi)

Required conda envs:
  • LivePortrait env  (whatever you named it)  — main streamlit app
  • seed-vc env                                — voice transformation
  • whisper env       (configurable via WHISPER_ENV_NAME) — transcription
"""

import os
import sys
import json
import time
import subprocess
import threading
import queue
import wave
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
import librosa
from transformers import HubertModel
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

# ── streamlit-webrtc ──────────────────────────────────────────────────────────

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, AudioProcessorBase
    import av
    _WEBRTC_OK = True
except ImportError:
    _WEBRTC_OK = False

# ── streamlit-cropper ─────────────────────────────────────────────────────────

try:
    from streamlit_cropper import st_cropper
    _CROPPER_OK = True
except ImportError:
    _CROPPER_OK = False

# ── Project imports ───────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.argument_config   import ArgumentConfig
from src.config.inference_config  import InferenceConfig
from src.config.crop_config       import CropConfig
from src.live_portrait_pipeline   import LivePortraitPipeline
from src.evaluation_pipeline      import EvaluationPipeline
from src.utils.face_guide         import FaceGuideProcessor
from src.utils.file_manager       import (
    ensure_dirs, save_upload, cleanup_old_sessions,
)
from src.utils.video_utils        import (
    get_video_duration, extract_frame,
    trim_video, crop_video,
    extract_audio, merge_audio_video, re_encode_for_browser,
)

# ── Constants ─────────────────────────────────────────────────────────────────

PREDEF_VOICES_DIR = os.path.abspath(
    os.path.join("..", "Voice-Transformation", "predefined_voices")
)
PREDEF_VOICE_MAP = {
    "Male — High Pitch":    "male_high.wav",
    "Male — Medium Pitch":  "male_medium.wav",
    "Male — Low Pitch":     "male_low.wav",
    "Female — High Pitch":  "female_high.wav",
    "Female — Medium Pitch":"female_medium.wav",
    "Female — Low Pitch":   "female_low.wav",
}

SEED_VC_REPO = os.path.abspath(os.path.join("..", "Voice-Transformation"))

# Whisper runs in its own conda env to avoid torch/numpy/numba conflicts with
# the LivePortrait env. Change this to whatever you named the env.
WHISPER_ENV_NAME = "whisper"
WHISPER_MODEL_SIZE = "base"

# UI label  →  Whisper language code ( None ⇒ auto-detect )
WHISPER_LANGUAGE_MAP = {
    "Auto-detect": None,
    "English":     "en",
    "Hindi":       "hi",
}

# Fixed output path for webcam recording (per session, overwritten on retry)
REC_PATH = os.path.join("temp_patients", "webcam_recording.mp4")

# Guardrail thresholds
MIN_VIDEO_DURATION_S        = 1.0
MAX_VIDEO_DURATION_S        = 60.0
MIN_FRAME_COUNT             = 10
SILENCE_RMS_THRESHOLD       = 1e-3
STATIONARY_DIFF_THRESHOLD   = 2.0   # mean per-pixel diff between sampled frames

# ── Final-score weights (three-way) ──────────────────────────────────────────
# These must sum to 1.0 when all three components are present.
WEIGHT_ARTICULATION   = 0.30
WEIGHT_DIALOGUE_ACOU  = 0.30   # ContentVec + FastDTW
WEIGHT_DIALOGUE_TRANS = 0.40   # Whisper + WER

# ── Startup ───────────────────────────────────────────────────────────────────

ensure_dirs()
cleanup_old_sessions(max_age_hours=2)

# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped temp-file cleanup
# ─────────────────────────────────────────────────────────────────────────────

class _SessionFiles:
    """Tracks all temp files written during this browser session."""

    def __init__(self):
        self._files: list[str] = []
        self._lock = threading.Lock()

    def add(self, path: str) -> str:
        if path:
            with self._lock:
                self._files.append(os.path.abspath(path))
        return path

    def cleanup(self):
        with self._lock:
            for f in self._files:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                except OSError:
                    pass
            self._files.clear()

    def __del__(self):
        self.cleanup()

def _sess() -> _SessionFiles:
    if "_session_files" not in st.session_state:
        st.session_state["_session_files"] = _SessionFiles()
    return st.session_state["_session_files"]

def _save(uploaded_file, save_dir: str = "temp_uploads") -> str | None:
    """Persist an uploaded file and register it for session cleanup."""
    path = save_upload(uploaded_file, save_dir=save_dir)
    if path:
        _sess().add(path)
    return path

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def _load_models():
    cfg_inf  = InferenceConfig()
    cfg_crop = CropConfig()
    gen = LivePortraitPipeline(inference_cfg=cfg_inf, crop_cfg=cfg_crop)
    evl = EvaluationPipeline (inference_cfg=cfg_inf, crop_cfg=cfg_crop)
    return gen, evl

# ─────────────────────────────────────────────────────────────────────────────
# WebRTC processors
# ─────────────────────────────────────────────────────────────────────────────

if _WEBRTC_OK:
    class _AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.audio_frames = []
            self.recording = False
            self.lock = threading.Lock()

        async def recv_queued(self, frames: list["av.AudioFrame"]) -> list["av.AudioFrame"]:
            with self.lock:
                if self.recording:
                    self.audio_frames.extend(frames)
            return frames

    class _GuidedRecorder(VideoProcessorBase):
        def __init__(self):
            self._guide     = FaceGuideProcessor()
            self.feedback_q: "queue.Queue[str]" = queue.Queue(maxsize=2)
            self._lock      = threading.Lock()
            self._recording = False
            self._writer: "cv2.VideoWriter | None" = None
            self._out_path: "str | None" = None
            self._frames_written = 0

            self._target_fps = 30.0
            self._frame_interval = 1.0 / self._target_fps
            self._next_expected_time: float | None = None

            self._audio_frames: list = []

        def start_recording(self, output_path: str) -> None:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            with self._lock:
                self._out_path       = output_path
                self._recording      = True
                self._writer         = None
                self._frames_written = 0
                self._next_expected_time = None
                self._audio_frames   = []

        def stop_recording(self) -> "str | None":
            with self._lock:
                self._recording = False
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                path = self._out_path
                self._out_path = None
                frames = self._frames_written
                audio_frames = list(self._audio_frames)
                self._audio_frames.clear()

            if frames == 0 and path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
                return None

            if audio_frames and path and os.path.isfile(path):
                audio_path = path.replace(".mp4", "_audio.wav")
                try:
                    if _save_recorded_audio(audio_frames, audio_path):
                        video_duration = frames / self._target_fps
                        with wave.open(audio_path, "rb") as wf:
                            audio_duration = wf.getnframes() / float(wf.getframerate())

                        merged_path = path.replace(".mp4", "_av.mp4")

                        if video_duration > 0 and audio_duration > 0:
                            speed_factor = audio_duration / video_duration
                            speed_factor = max(0.5, min(100.0, speed_factor))

                            cmd = [
                                "ffmpeg", "-y",
                                "-i", path,
                                "-i", audio_path,
                                "-map", "0:v",
                                "-map", "1:a",
                                "-c:v", "copy",
                                "-c:a", "aac",
                                "-shortest",
                                merged_path
                            ]

                            subprocess.run(
                                cmd,
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )

                            for tmp in (audio_path, path):
                                try:
                                    os.remove(tmp)
                                except OSError:
                                    pass

                            return merged_path
                except Exception as e:
                    print(f"Sync/Merge error: {e}")
                finally:
                    if os.path.isfile(audio_path):
                        try:
                            os.remove(audio_path)
                        except OSError:
                            pass

            return path

        def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
            raw_bgr = frame.to_ndarray(format="bgr24")
            h, w    = raw_bgr.shape[:2]

            annotated, feedback = self._guide.process_frame(raw_bgr)

            try:
                self.feedback_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.feedback_q.put_nowait(feedback)
            except queue.Full:
                pass

            ts = frame.time if frame.time is not None else time.time()

            with self._lock:
                if self._recording and self._out_path:
                    if self._writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._writer = cv2.VideoWriter(
                            self._out_path, fourcc, self._target_fps, (w, h)
                        )
                        self._next_expected_time = ts

                    if self._writer.isOpened():
                        duplications = 0
                        while self._next_expected_time <= ts and duplications < 15:
                            self._writer.write(raw_bgr)
                            self._frames_written += 1
                            self._next_expected_time += self._frame_interval
                            duplications += 1

                        if self._next_expected_time < ts:
                            self._next_expected_time = ts

                    is_rec = True
                else:
                    is_rec = False

            display = annotated if annotated is not None else raw_bgr.copy()
            if is_rec:
                cv2.circle(display, (28, 28), 11, (0, 0, 220), -1, cv2.LINE_AA)
                cv2.putText(
                    display, "REC", (46, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 220), 2, cv2.LINE_AA,
                )

            return av.VideoFrame.from_ndarray(display, format="bgr24")

# ─────────────────────────────────────────────────────────────────────────────
# Audio recording helper
# ─────────────────────────────────────────────────────────────────────────────

def _write_wav(path: str, audio_np: np.ndarray, sample_rate: int):
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def _save_recorded_audio(frames: list, output_path: str) -> bool:
    if not frames:
        return False

    try:
        sample_rate = frames[0].sample_rate
        audio_arrays = []

        for f in frames:
            arr = f.to_ndarray()
            if arr.ndim == 2 and arr.shape[0] == 1:
                packed = arr[0]
                mono = packed[::2]
            else:
                mono = arr.squeeze()
            mono = mono.astype(np.float32) / 32768.0
            audio_arrays.append(mono)

        audio_raw = np.concatenate(audio_arrays)
        peak = np.max(np.abs(audio_raw)) + 1e-8
        audio_norm = np.clip(audio_raw / peak, -1.0, 1.0)
        _write_wav(output_path, audio_norm, sample_rate)
        return True

    except Exception as e:
        print("Audio save error:", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation guardrails
# ─────────────────────────────────────────────────────────────────────────────

def _is_video_stationary(video_path: str,
                         sample_frames: int = 30,
                         threshold: float = STATIONARY_DIFF_THRESHOLD) -> bool:
    """
    Quick stationary-frame detector using mean absolute pixel diff between
    evenly-spaced sampled frames (downscaled for speed).
    Returns True if the video is essentially still.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return True

    indices = np.linspace(0, total_frames - 1,
                          min(sample_frames, total_frames)).astype(int)

    prev_gray = None
    diffs = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            diffs.append(diff)
        prev_gray = gray

    cap.release()

    if not diffs:
        return True

    return float(np.mean(diffs)) < threshold


def _audio_has_speech(audio_path: str,
                      rms_threshold: float = SILENCE_RMS_THRESHOLD
                      ) -> tuple[bool, float]:
    """
    Returns (has_speech, observed_rms).
    """
    try:
        data, _ = librosa.load(audio_path, sr=16000, mono=True)
        if len(data) == 0:
            return False, 0.0
        rms = float(np.sqrt(np.mean(data ** 2)))
        return rms >= rms_threshold, rms
    except Exception:
        return False, 0.0


def _validate_video_for_evaluation(video_path: str,
                                   label: str = "video") -> list[str]:
    """
    Battery of guardrail checks. Returns list of error messages — empty list
    means the video passed all checks.
    """
    errors: list[str] = []

    if not video_path or not os.path.isfile(video_path):
        errors.append(f"{label}: file not found.")
        return errors

    # Duration
    try:
        duration = float(get_video_duration(video_path))
    except Exception:
        errors.append(f"{label}: could not read video duration.")
        return errors

    if duration < MIN_VIDEO_DURATION_S:
        errors.append(
            f"{label}: too short ({duration:.1f}s). Minimum is {MIN_VIDEO_DURATION_S}s."
        )
    if duration > MAX_VIDEO_DURATION_S:
        errors.append(
            f"{label}: too long ({duration:.1f}s). Please trim under {MAX_VIDEO_DURATION_S}s."
        )

    # Frame count
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count < MIN_FRAME_COUNT:
        errors.append(
            f"{label}: only {frame_count} frames decoded. Minimum is {MIN_FRAME_COUNT}."
        )

    # Stationary check
    if frame_count >= MIN_FRAME_COUNT:
        try:
            if _is_video_stationary(video_path):
                errors.append(
                    f"{label}: appears stationary (no motion detected). "
                    "Make sure the camera and subject are moving naturally."
                )
        except Exception:
            pass

    # Audio presence + non-silence
    tmp_audio = os.path.join(
        "temp_uploads",
        f"_validate_{label.replace(' ', '_')}_{int(time.time()*1000)}.wav"
    )
    try:
        ok = extract_audio(video_path, tmp_audio)
        if not ok or not os.path.isfile(tmp_audio):
            errors.append(f"{label}: no audio track found.")
        else:
            has_speech, rms = _audio_has_speech(tmp_audio)
            if not has_speech:
                errors.append(
                    f"{label}: audio is silent or below speech-energy "
                    f"threshold (RMS={rms:.4f}, need ≥{SILENCE_RMS_THRESHOLD}). "
                    "Please re-record with clearer audio."
                )
    except Exception as e:
        errors.append(f"{label}: audio validation failed ({e}).")
    finally:
        if os.path.isfile(tmp_audio):
            try:
                os.remove(tmp_audio)
            except OSError:
                pass

    return errors


# ═════════════════════════════════════════════════════════════════════════════
# DIALOGUE ACCURACY 1  —  Acoustic similarity
#   silero-vad → ContentVec embeddings → FastDTW (cosine) → linear score
# ═════════════════════════════════════════════════════════════════════════════

_CONTENTVEC_MODEL = None
_CONTENTVEC_DEVICE = None
_VAD_MODEL = None
_VAD_UTILS = None


def _get_contentvec():
    global _CONTENTVEC_MODEL, _CONTENTVEC_DEVICE
    if _CONTENTVEC_MODEL is None:
        _CONTENTVEC_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # HF port of ContentVec-768 (loads with the HuBERT architecture)
        model = HubertModel.from_pretrained("lengyue233/content-vec-best")
        model.to(_CONTENTVEC_DEVICE).eval()
        _CONTENTVEC_MODEL = model
    return _CONTENTVEC_MODEL, _CONTENTVEC_DEVICE


def _get_vad():
    """Load silero-vad model + utils from torch.hub (cached after first call)."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
        )
        _VAD_MODEL = model
        _VAD_UTILS = utils
    return _VAD_MODEL, _VAD_UTILS


def _apply_vad(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    label: str = "",
) -> np.ndarray:
    """
    Run silero-vad and return only the concatenated speech regions.
    Falls back to the original audio if no speech is detected.
    """
    model, utils = _get_vad()
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
    )

    if not timestamps:
        print(f"  [VAD {label}] No speech detected — using original audio.")
        return audio

    speech_chunks = [audio[t['start']:t['end']] for t in timestamps]
    speech_audio = np.concatenate(speech_chunks)

    orig_dur = len(audio) / sample_rate
    new_dur  = len(speech_audio) / sample_rate
    print(
        f"  [VAD {label}] {orig_dur:.2f}s → {new_dur:.2f}s "
        f"({len(timestamps)} segments kept)"
    )

    return speech_audio.astype(np.float32)


def _compute_dialogue_accuracy_acoustic(
    ref_video: str,
    pat_video: str,
    use_vad: bool = True,
    vad_threshold: float = 0.5,
    fastdtw_radius: int = 1,
) -> "float | None":
    """
    Compare the audio content of the reference and patient videos using
    silero-vad pre-filtering, ContentVec embeddings, FastDTW alignment with
    cosine distance, and a softer linear score mapping.

    Score mapping:
        avg cosine distance 0.0 (perfect match)   → 100
        avg cosine distance 1.0 (unrelated)       →  50
        avg cosine distance 2.0 (anti-correlated) →   0

    Returns a 0–100 score, or None if audio cannot be extracted/processed.
    """
    ref_audio = os.path.join("temp_uploads", "_eval_ref_audio_acoustic.wav")
    pat_audio = os.path.join("temp_uploads", "_eval_pat_audio_acoustic.wav")

    try:
        if not extract_audio(ref_video, ref_audio):
            return None
        if not extract_audio(pat_video, pat_audio):
            return None

        TARGET_SR = 16000  # ContentVec is trained on 16 kHz mono

        # ── Load + downmix + resample to 16 kHz ────────────────────────
        ref_data, _ = librosa.load(ref_audio, sr=TARGET_SR, mono=True)
        pat_data, _ = librosa.load(pat_audio, sr=TARGET_SR, mono=True)

        # Need at least ~0.1 s of audio in each
        if len(ref_data) < 1600 or len(pat_data) < 1600:
            return None

        ref_data = ref_data.astype(np.float32)
        pat_data = pat_data.astype(np.float32)

        # ── VAD: strip silence / non-speech from both clips ────────────
        if use_vad:
            ref_data = _apply_vad(
                ref_data, sample_rate=TARGET_SR,
                threshold=vad_threshold, label="ref",
            )
            pat_data = _apply_vad(
                pat_data, sample_rate=TARGET_SR,
                threshold=vad_threshold, label="pat",
            )

            if len(ref_data) < 1600 or len(pat_data) < 1600:
                return None

        # ── ContentVec frame-level embeddings (T × 768) ────────────────
        model, device = _get_contentvec()

        @torch.no_grad()
        def _embed(audio: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            feats = model(x).last_hidden_state.squeeze(0).cpu().numpy()
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            return (feats / norms).astype(np.float64)

        ref_feat = _embed(ref_data)
        pat_feat = _embed(pat_data)

        if len(ref_feat) == 0 or len(pat_feat) == 0:
            return None

        # ── FastDTW with cosine distance ───────────────────────────────
        distance, path = fastdtw(
            ref_feat, pat_feat,
            radius=fastdtw_radius,
            dist=cosine,
        )

        avg_cost = distance / len(path)       # ∈ [0, 2]
        mean_sim = 2.0 - avg_cost             # ∈ [0, 2]
        score    = max(0.0, mean_sim) * 50.0
        return min(100.0, max(0.0, score))

    except Exception as e:
        import traceback
        st.error(f"Acoustic dialogue accuracy failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    finally:
        for p in (ref_audio, pat_audio):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# DIALOGUE ACCURACY 2  —  Transcript similarity (Whisper + WER)
# ═════════════════════════════════════════════════════════════════════════════

def _run_whisper_subprocess(ref_audio: str,
                            pat_audio: str,
                            model_size: str = WHISPER_MODEL_SIZE,
                            language: "str | None" = None,
                            ) -> tuple["tuple[str, str] | None", "str | None"]:
    """
    Transcribe two audio files using Whisper via a subprocess that activates
    the dedicated `WHISPER_ENV_NAME` conda env. Returns ((ref_text, pat_text),
    error). On success, error is None. Both texts are lower-cased and stripped.

    `language` is a Whisper language code (e.g. "en", "hi") or None for
    auto-detection.
    """
    ref_audio   = os.path.abspath(ref_audio)
    pat_audio   = os.path.abspath(pat_audio)
    script_path = os.path.abspath(os.path.join("temp_uploads", "run_whisper.py"))
    bat_path    = os.path.abspath(os.path.join("temp_uploads", "run_whisper.bat"))
    output_json = os.path.abspath(os.path.join("temp_uploads", "whisper_output.json"))

    if os.path.exists(output_json):
        try:
            os.remove(output_json)
        except OSError:
            pass

    # Build the language kwarg snippet. When None ⇒ omit it so Whisper
    # falls back to its built-in language identification.
    lang_kwarg = f', language="{language}"' if language else ""

    script_body = f'''import json, sys, traceback
out = {{"ref_text": "", "pat_text": "", "error": None, "ref_lang": "", "pat_lang": ""}}
try:
    import whisper, torch
    model = whisper.load_model("{model_size}")
    fp16  = torch.cuda.is_available()
    ref = model.transcribe(r"{ref_audio}", fp16=fp16{lang_kwarg})
    pat = model.transcribe(r"{pat_audio}", fp16=fp16{lang_kwarg})
    out["ref_text"] = (ref.get("text") or "").strip().lower()
    out["pat_text"] = (pat.get("text") or "").strip().lower()
    out["ref_lang"] = ref.get("language", "") or ""
    out["pat_lang"] = pat.get("language", "") or ""
except Exception as e:
    out["error"] = f"{{type(e).__name__}}: {{e}}\\n{{traceback.format_exc()}}"

with open(r"{output_json}", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_body)

    bat_body = (
        '@echo off\n'
        'call "%UserProfile%\\Miniconda3\\Scripts\\activate.bat"\n'
        f'call conda activate {WHISPER_ENV_NAME}\n'
        f'python "{script_path}"\n'
    )
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_body)

    try:
        subprocess.run(bat_path, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return None, (
            f"Whisper subprocess exited with non-zero status ({e.returncode}). "
            f"Check the console output above for details. "
            f"Verify the conda env '{WHISPER_ENV_NAME}' exists and has "
            "openai-whisper installed."
        )
    except Exception as e:
        return None, f"Failed to launch whisper subprocess: {type(e).__name__}: {e}"

    if not os.path.exists(output_json):
        return None, (
            f"Whisper subprocess did not produce {output_json}. "
            "The .bat probably failed silently — check the console."
        )

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception as e:
        return None, f"Failed to parse whisper output JSON: {type(e).__name__}: {e}"

    if result.get("error"):
        return None, result["error"]

    ref_text = result.get("ref_text", "")
    pat_text = result.get("pat_text", "")
    ref_lang = result.get("ref_lang", "")
    pat_lang = result.get("pat_lang", "")

    if not ref_text and not pat_text:
        return None, "Whisper returned empty text for both clips (audio may be unintelligible)."

    # Print detected/forced language to console for debugging
    print(f"[Whisper] ref_lang={ref_lang!r}  pat_lang={pat_lang!r}  "
          f"(forced={language!r})")

    return (ref_text, pat_text), None


def _safe_wer(ref: str, hyp: str) -> float:
    """
    Word Error Rate via word-level Levenshtein distance. Returns a float
    in [0.0, 1.0] (clamped at 1.0 for very bad transcripts).
    """
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    if not hyp_words:
        return 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    return min(1.0, dp[n][m] / n)


def _highlight_word_diff(ref: str, hyp: str) -> str:
    """
    Naïve position-aligned word diff. Green for matches, red for mismatches.
    """
    r = ref.split()
    h = hyp.split()
    out = []

    for i in range(max(len(r), len(h))):
        rw = r[i] if i < len(r) else ""
        hw = h[i] if i < len(h) else ""
        if not hw:
            out.append(f"<span style='color:#dc2626;text-decoration:line-through;'>{rw}</span>")
        elif rw == hw:
            out.append(f"<span style='color:#16a34a;'>{hw}</span>")
        else:
            out.append(f"<span style='color:#dc2626;font-weight:600;'>{hw}</span>")

    return " ".join(out)


def _compute_dialogue_accuracy_transcript(
    ref_video: str,
    pat_video: str,
    language: "str | None" = None,
) -> tuple["float | None", str, str, "str | None"]:
    """
    Whisper-based dialogue accuracy.

    `language` is a Whisper language code (e.g. "en", "hi") or None for
    auto-detection.

    Returns (score_0_to_100, ref_text, pat_text, reason).
    On success: reason is None.
    On failure: score is None, reason explains what went wrong.
    """
    ref_audio = os.path.join("temp_uploads", "_eval_ref_audio_trans.wav")
    pat_audio = os.path.join("temp_uploads", "_eval_pat_audio_trans.wav")

    try:
        if not extract_audio(ref_video, ref_audio):
            return None, "", "", "Failed to extract audio from the driving (reference) video."
        if not extract_audio(pat_video, pat_audio):
            return None, "", "", "Failed to extract audio from the patient video."

        ref_ok, ref_rms = _audio_has_speech(ref_audio)
        pat_ok, pat_rms = _audio_has_speech(pat_audio)
        if not ref_ok:
            return (None, "", "",
                    f"Reference audio is too quiet (RMS={ref_rms:.4f}, "
                    f"need ≥{SILENCE_RMS_THRESHOLD}).")
        if not pat_ok:
            return (None, "", "",
                    f"Patient audio is too quiet (RMS={pat_rms:.4f}, "
                    f"need ≥{SILENCE_RMS_THRESHOLD}).")

        texts, err = _run_whisper_subprocess(
            ref_audio, pat_audio, language=language,
        )
        if err is not None or texts is None:
            return None, "", "", f"Whisper transcription failed: {err}"

        ref_text, pat_text = texts
        if not ref_text:
            return None, ref_text, pat_text, "Reference transcription returned empty text."
        if not pat_text:
            return None, ref_text, pat_text, "Patient transcription returned empty text."

        wer_score = _safe_wer(ref_text, pat_text)
        score = max(0.0, min(100.0, (1.0 - wer_score) * 100.0))
        return score, ref_text, pat_text, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, "", "", f"Unexpected error: {type(e).__name__}: {e}"
    finally:
        for p in (ref_audio, pat_audio):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# Final weighted score (three components, graceful fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_weighted_final_score(
    a_score: float,
    d_acoustic: "float | None",
    d_transcript: "float | None",
) -> tuple[int, str]:
    """
    Weighted combination of articulation + both dialogue accuracies.

    Defaults: 30 % articulation, 30 % acoustic dialogue, 40 % transcript dialogue.

    Fallback policy:
      • Both dialogue scores present  → use full three-way weights.
      • Only acoustic available       → 40 % articulation + 60 % acoustic.
      • Only transcript available     → 40 % articulation + 60 % transcript.
      • Neither available             → 100 % articulation.

    Penalty:
      If both dialogue scores are present and their average is < 40, the
      articulation contribution is halved (saying the wrong thing should not
      be rewarded just because the lips moved well).

    Returns (final_score_int_0_to_100, human_readable_weight_note).
    """
    if d_acoustic is not None and d_transcript is not None:
        avg_dialogue = 0.5 * (d_acoustic + d_transcript)
        if avg_dialogue < 40:
            # Halve articulation contribution; redistribute nothing — caller
            # already penalises by lower dialogue score itself.
            final = (
                (WEIGHT_ARTICULATION * 0.5) * a_score
                + WEIGHT_DIALOGUE_ACOU       * d_acoustic
                + WEIGHT_DIALOGUE_TRANS      * d_transcript
            )
            note = (
                f"{int(WEIGHT_ARTICULATION*50)} % Articulation (penalised) + "
                f"{int(WEIGHT_DIALOGUE_ACOU*100)} % Acoustic + "
                f"{int(WEIGHT_DIALOGUE_TRANS*100)} % Transcript"
            )
        else:
            final = (
                WEIGHT_ARTICULATION   * a_score
                + WEIGHT_DIALOGUE_ACOU  * d_acoustic
                + WEIGHT_DIALOGUE_TRANS * d_transcript
            )
            note = (
                f"{int(WEIGHT_ARTICULATION*100)} % Articulation + "
                f"{int(WEIGHT_DIALOGUE_ACOU*100)} % Acoustic + "
                f"{int(WEIGHT_DIALOGUE_TRANS*100)} % Transcript"
            )
        return int(round(max(0.0, min(100.0, final)))), note

    # Two-way fallbacks
    if d_acoustic is not None and d_transcript is None:
        final = 0.4 * a_score + 0.6 * d_acoustic
        return int(round(max(0.0, min(100.0, final)))), (
            "40 % Articulation + 60 % Acoustic (Transcript unavailable)"
        )

    if d_transcript is not None and d_acoustic is None:
        final = 0.4 * a_score + 0.6 * d_transcript
        return int(round(max(0.0, min(100.0, final)))), (
            "40 % Articulation + 60 % Transcript (Acoustic unavailable)"
        )

    # Articulation only
    return int(round(max(0.0, min(100.0, a_score)))), (
        "100 % Articulation (both dialogue methods unavailable)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Voice transformation (local Seed-VC V2)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_voice_transformation(gen_video: str, ref_voice_path: str) -> str:
    source_audio = os.path.abspath(os.path.join("temp_uploads", "extracted_source.wav"))
    output_audio = os.path.abspath(os.path.join("temp_uploads", "transformed_audio.wav"))
    repo_vc      = os.path.abspath(os.path.join("..", "Voice-Transformation"))
    script_path  = os.path.abspath(os.path.join("temp_uploads", "run_vc.py"))
    bat_path     = os.path.abspath(os.path.join("temp_uploads", "run_vc.bat"))

    if os.path.exists(source_audio):
        os.remove(source_audio)

    if not extract_audio(gen_video, source_audio):
        st.warning("No audio track in driving video — skipping voice transformation.")
        return gen_video

    script_body = f'''import sys, os, scipy.io.wavfile as wavfile
sys.path.append(r"{repo_vc}")
os.chdir(r"{repo_vc}")
from argparse import Namespace
import app

args = Namespace(compile=False, enable_v1=False, enable_v2=True)
app.vc_wrapper_v2 = app.load_v2_models(args)

gen = app.convert_voice_v2_wrapper(
    source_audio_path        = r"{source_audio}",
    target_audio_path        = r"{os.path.abspath(ref_voice_path)}",
    diffusion_steps          = 30,
    length_adjust            = 1.0,
    intelligebility_cfg_rate = 0.0,
    similarity_cfg_rate      = 0.7,
    top_p                    = 0.9,
    temperature              = 1.0,
    repetition_penalty       = 1.0,
    convert_style            = False,
    anonymization_only       = False,
    stream_output            = True,
)
outputs = list(gen)
if not outputs:
    print("ERROR: no outputs"); sys.exit(1)
final = outputs[-1]
if isinstance(final, tuple) and len(final) == 2:
    _, full_audio = final
else:
    full_audio = final
if isinstance(full_audio, tuple) and len(full_audio) == 2:
    sr, data = full_audio
    wavfile.write(r"{output_audio}", sr, data)
else:
    print("ERROR: unexpected structure:", type(full_audio)); sys.exit(1)
'''
    with open(script_path, "w") as f:
        f.write(script_body)

    bat_body = (
        '@echo off\n'
        'call "%UserProfile%\\Miniconda3\\Scripts\\activate.bat"\n'
        'call conda activate seed-vc\n'
        f'python "{script_path}"\n'
    )
    with open(bat_path, "w") as f:
        f.write(bat_body)

    try:
        subprocess.run(bat_path, shell=True, check=True)
    except subprocess.CalledProcessError:
        st.error("Voice transformation subprocess failed — returning standard reenactment.")
        return gen_video

    out_dir        = "animations"
    final_vid_path = os.path.join(out_dir, "final_voice_" + os.path.basename(gen_video))
    if merge_audio_video(gen_video, output_audio, final_vid_path):
        return final_vid_path
    return gen_video


# ─────────────────────────────────────────────────────────────────────────────
# Webcam recording panel
# ─────────────────────────────────────────────────────────────────────────────

def _set_rec_state(action: str, state: str):
    """Callback triggered natively by UI to preempt double-rerun flashing."""
    st.session_state["rec_action"] = action
    st.session_state["rec_state"]  = state

def _render_recording_panel() -> "str | None":
    if not _WEBRTC_OK:
        st.error(
            "Webcam recording requires `streamlit-webrtc`. "
            "Install:  `pip install streamlit-webrtc av aiortc`"
        )
        return None

    for key, default in [
        ("rec_state",      "idle"),
        ("rec_saved_path", None),
        ("rec_action",     None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    ctx = webrtc_streamer(
        key                      = "guided-recorder",
        mode                     = WebRtcMode.SENDRECV,
        video_processor_factory  = _GuidedRecorder,
        audio_processor_factory  = _AudioRecorder,
        media_stream_constraints = {
            "video": True,
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            },
        },
        async_processing         = True,
    )

    action = st.session_state["rec_action"]

    if action == "start":
        if ctx.video_processor:
            _sess().add(REC_PATH)
            ctx.video_processor.start_recording(REC_PATH)
        if ctx.audio_processor:
            with ctx.audio_processor.lock:
                ctx.audio_processor.recording = True
                ctx.audio_processor.audio_frames = []
        st.session_state["rec_action"] = None

    elif action == "stop":
        if ctx.audio_processor:
            with ctx.audio_processor.lock:
                ctx.audio_processor.recording = False
                audio_frames = list(ctx.audio_processor.audio_frames)
                ctx.audio_processor.audio_frames.clear()

            if ctx.video_processor:
                with ctx.video_processor._lock:
                    ctx.video_processor._audio_frames = audio_frames

        if ctx.video_processor:
            saved = ctx.video_processor.stop_recording()
        else:
            saved = REC_PATH if os.path.isfile(REC_PATH) and os.path.getsize(REC_PATH) > 1024 else None

        if saved and os.path.isfile(saved):
            with st.spinner("Processing recording..."):
                saved = re_encode_for_browser(saved)
                _sess().add(saved)
                st.session_state["rec_saved_path"] = saved
        else:
            st.warning("No valid frames captured — please try again.")
            st.session_state["rec_state"] = "idle"

        st.session_state["rec_action"] = None

    elif action == "retry":
        saved_path = st.session_state.get("rec_saved_path")
        if saved_path and os.path.isfile(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
        if ctx.video_processor:
            ctx.video_processor.stop_recording()
        if ctx.audio_processor:
            with ctx.audio_processor.lock:
                ctx.audio_processor.recording = False
                ctx.audio_processor.audio_frames.clear()

        st.session_state["rec_saved_path"] = None
        st.session_state["rec_action"]     = None

    rec_state = st.session_state["rec_state"]

    fb_spot = st.empty()
    if ctx.video_processor:
        try:
            fb = ctx.video_processor.feedback_q.get_nowait()
            good   = "✓" in fb or "Perfect" in fb
            colour = "#22c55e" if good else "#f97316"
            fb_spot.markdown(
                f'<p style="color:{colour};font-weight:700;font-size:1rem;">{fb}</p>',
                unsafe_allow_html=True,
            )
        except queue.Empty:
            pass

    st.divider()

    if rec_state == "idle":
        st.markdown(
            "**① Center your face** in the guide above until the oval turns "
            "**green**, then click **Start Recording**."
        )

        cam_ready = ctx.state.playing
        if not cam_ready:
            st.info("Click **START** in the webcam widget above to turn on your camera.")

        st.button(
            "🔴  Start Recording",
            type                = "primary",
            use_container_width = True,
            disabled            = not cam_ready,
            on_click            = _set_rec_state,
            args                = ("start", "recording")
        )

    elif rec_state == "recording":
        st.markdown(
            '<p style="color:#ef4444;font-weight:700;font-size:1.05rem;">'
            '● Recording in progress…'
            '</p>',
            unsafe_allow_html=True,
        )
        st.button(
            "⏹  Stop Recording",
            type                = "primary",
            use_container_width = True,
            on_click            = _set_rec_state,
            args                = ("stop", "preview")
        )

    elif rec_state == "preview":
        saved_path = st.session_state.get("rec_saved_path")

        if not (saved_path and os.path.isfile(saved_path)):
            st.warning("Saved recording is missing — please record again.")
            st.button(
                "Reset Interface",
                on_click = _set_rec_state,
                args     = ("retry", "idle")
            )
            return None

        st.success("✅  Recording saved — preview:")
        st.video(saved_path)

        c_use, c_retry = st.columns(2)

        with c_use:
            if st.button(
                "✅  Use This Recording",
                type                = "primary",
                use_container_width = True,
            ):
                return saved_path

        with c_retry:
            st.button(
                "🔄  Record Again",
                use_container_width = True,
                on_click            = _set_rec_state,
                args                = ("retry", "idle")
            )

        return saved_path

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(layout="wide", page_title="LivePortrait – Speech Therapy Studio")

    st.markdown(
        """
        <h1 style="
          background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          font-size:2rem;font-weight:800;margin-bottom:0;">
          🗣️ LivePortrait · Speech Therapy Studio
        </h1>
        <p style="color:#64748b;margin-top:4px;">
          Animate a reference face · transform voice · evaluate patient articulation
        </p>
        """,
        unsafe_allow_html=True,
    )

    _DEFAULTS = {
        "src_upload_name": None,
        "src_path":        None,
        "img_is_cropped":  False,
        "drv_upload_name": None,
        "drv_path":        None,
        "vid_is_cropped":  False,
        "generated_video": None,
        "pat_upload_name": None,
        "pat_upload_path": None,
        "_active_pat_path": None,
        "pat_is_cropped":   False,
        "pat_final_path":   None,
    }
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    generator, evaluator = _load_models()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️  Settings")
        flag_do_crop         = st.checkbox("Auto Crop Face",     value=True)
        flag_paste_back      = st.checkbox("Paste Back",         value=True)
        flag_normalize_lip   = st.checkbox("Neutral Lip Start",  value=True)
        flag_stitching       = st.checkbox("Seamless Stitching", value=True)
        flag_relative_motion = st.checkbox("Relative Motion",    value=True)
        driving_option       = st.selectbox(
            "Driving Mode", ["expression-friendly", "pose-friendly"], index=0
        )

        st.divider()
        st.subheader("🗣️  Evaluation")
        whisper_lang_label = st.selectbox(
            "Transcription Language",
            list(WHISPER_LANGUAGE_MAP.keys()),
            index=0,
            help=(
                "Language used by Whisper for the transcript-based dialogue "
                "accuracy score. Choose **Auto-detect** if you're unsure — "
                "Whisper will identify the language automatically. Force a "
                "specific language (English / Hindi) for short or noisy clips "
                "where auto-detect can pick the wrong one."
            ),
        )

        st.divider()
        if st.button("🗑️  Clear temp files"):
            _sess().cleanup()
            st.success("Temp files removed.")

    # Resolve the language label → Whisper code (None ⇒ auto-detect)
    whisper_lang_code = WHISPER_LANGUAGE_MAP[whisper_lang_label]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 – SOURCE IMAGE
    # ═══════════════════════════════════════════════════════════════════════
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Source Identity")
        source_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if source_file:
            if st.session_state["src_upload_name"] != source_file.name:
                st.session_state["src_upload_name"] = source_file.name
                st.session_state["src_path"]        = _save(source_file)
                st.session_state["img_is_cropped"]  = False

            if not st.session_state["img_is_cropped"]:
                if _CROPPER_OK:
                    img = Image.open(st.session_state["src_path"])
                    st.info("Adjust the white box to frame the face, then click Apply.")
                    cropped_img = st_cropper(
                        img,
                        realtime_update = True,
                        box_color       = "#FFFFFF",
                        aspect_ratio    = (1, 1),
                        key             = "src_cropper",
                    )
                    if st.button("✂️  Apply Image Crop", use_container_width=True):
                        save_path = os.path.join(
                            "temp_uploads", f"cropped_{source_file.name}"
                        )
                        cropped_img.save(save_path)
                        _sess().add(save_path)
                        st.session_state["src_path"]       = save_path
                        st.session_state["img_is_cropped"] = True
                        st.rerun()
                else:
                    st.warning(
                        "Install `streamlit-cropper` for manual crop. "
                        "LivePortrait will still auto-detect the face."
                    )
                    st.image(st.session_state["src_path"], caption="Source image")
                    st.session_state["img_is_cropped"] = True

            else:
                st.image(
                    st.session_state["src_path"],
                    caption = "✅  Processed source",
                )
                if st.button("🔄  Reset Image", use_container_width=True):
                    st.session_state["src_upload_name"] = None
                    st.session_state["src_path"]        = _save(source_file)
                    st.session_state["img_is_cropped"]  = False
                    st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 – DRIVING VIDEO
    # ═══════════════════════════════════════════════════════════════════════
    with col2:
        st.subheader("2. Driving Motion")
        driving_file = st.file_uploader(
            "Upload Driving Video", type=["mp4", "mov", "avi"]
        )

        if driving_file:
            if st.session_state["drv_upload_name"] != driving_file.name:
                st.session_state["drv_upload_name"] = driving_file.name
                st.session_state["drv_path"]        = _save(driving_file)
                st.session_state["vid_is_cropped"]  = False

            dur = float(get_video_duration(st.session_state["drv_path"]))
            tc1, tc2 = st.columns(2)
            with tc1:
                trim_start = st.slider("Trim start (s)", 0.0, dur, 0.0, 0.1)
            with tc2:
                trim_end   = st.slider("Trim end (s)",   0.0, dur, dur, 0.1)

            if not st.session_state["vid_is_cropped"]:
                frame = extract_frame(st.session_state["drv_path"], second=0)

                if frame and _CROPPER_OK:
                    st.info("Crop the face region using the first frame, then click Apply.")
                    crop_box = st_cropper(
                        frame,
                        realtime_update = True,
                        box_color       = "#FFFFFF",
                        aspect_ratio    = (1, 1),
                        return_type     = "box",
                        key             = "vid_cropper",
                    )
                if st.button("✂️  Apply Video Crop", use_container_width=True):
                    with st.spinner("Trimming, cropping & preserving audio…"):
                        trimmed_path = trim_video(
                            st.session_state["drv_path"], trim_start, trim_end
                        )

                        cropped_path = crop_video(
                            trimmed_path,
                            crop_box["left"], crop_box["top"],
                            crop_box["width"], crop_box["height"],
                        )

                        temp_audio = trimmed_path.replace(".mp4", "_temp_audio.wav")
                        if extract_audio(trimmed_path, temp_audio):
                            merged_path = cropped_path.replace(".mp4", "_with_audio.mp4")
                            if merge_audio_video(cropped_path, temp_audio, merged_path):
                                cropped_path = merged_path
                            try:
                                os.remove(temp_audio)
                            except OSError:
                                pass

                        _sess().add(cropped_path)
                        st.session_state["drv_path"]        = cropped_path
                        st.session_state["vid_is_cropped"]  = True
                        st.rerun()
                else:
                    if st.button("✂️  Apply Trim Only", use_container_width=True):
                        with st.spinner("Trimming…"):
                            path = trim_video(
                                st.session_state["drv_path"], trim_start, trim_end
                            )
                            _sess().add(path)
                            st.session_state["drv_path"]       = path
                            st.session_state["vid_is_cropped"] = True
                            st.rerun()
            else:
                st.video(st.session_state["drv_path"])
                if st.button("🔄  Reset Video", use_container_width=True):
                    st.session_state["drv_upload_name"] = None
                    st.session_state["drv_path"]        = _save(driving_file)
                    st.session_state["vid_is_cropped"]  = False
                    st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 – VOICE TRANSFORMATION
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("3. Voice Transformation (Optional)")

    voice_option = st.radio(
        "Select Voice Processing Mode:",
        ["Keep Original Voice", "Upload Reference Voice", "Predefined Voice Samples"],
        horizontal=True,
    )

    ref_voice_path = None

    if voice_option == "Upload Reference Voice":
        uploaded_voice = st.file_uploader(
            "Upload Target Voice File", type=["wav", "mp3", "m4a", "flac"]
        )
        if uploaded_voice:
            ref_voice_path = _save(uploaded_voice)

    elif voice_option == "Predefined Voice Samples":
        vc1, _ = st.columns(2)
        with vc1:
            predef_choice = st.selectbox(
                "Select Target Pitch / Gender:", list(PREDEF_VOICE_MAP.keys())
            )
        expected_path = os.path.join(PREDEF_VOICES_DIR, PREDEF_VOICE_MAP[predef_choice])
        os.makedirs(PREDEF_VOICES_DIR, exist_ok=True)
        if os.path.exists(expected_path):
            ref_voice_path = expected_path
            st.success(f"Loaded preset: {predef_choice}")
        else:
            st.warning(
                f"File not found: `{expected_path}`\n\n"
                "Place `.wav` files in `../Voice-Transformation/predefined_voices/`"
            )

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 – GENERATE
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    if st.session_state["src_path"] and st.session_state["drv_path"]:
        if st.button(
            "🚀  Generate Integrated Output", type="primary", use_container_width=True
        ):
            out_dir = "animations"
            args = ArgumentConfig(
                source               = st.session_state["src_path"],
                driving              = st.session_state["drv_path"],
                output_dir           = out_dir,
                flag_do_crop         = flag_do_crop,
                flag_pasteback       = flag_paste_back,
                flag_normalize_lip   = flag_normalize_lip,
                flag_stitching       = flag_stitching,
                flag_relative_motion = flag_relative_motion,
                driving_option       = driving_option,
            )

            with st.spinner("Running face reenactment…"):
                generator.execute(args)
                mp4_files = sorted(
                    [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                     if f.endswith(".mp4")],
                    key=os.path.getmtime,
                )

            if not mp4_files:
                st.error("Generation produced no output file.")
            else:
                gen_video = mp4_files[-1]

                if voice_option != "Keep Original Voice" and ref_voice_path:
                    with st.spinner("Applying voice transformation…"):
                        gen_video = _apply_voice_transformation(gen_video, ref_voice_path)

                with st.spinner("Encoding for browser…"):
                    gen_video = re_encode_for_browser(gen_video)

                st.session_state["generated_video"] = gen_video
                st.rerun()
    else:
        st.info("Upload a source image and a driving video to enable generation.")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5 – RESULTS & EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    if not st.session_state["generated_video"]:
        return

    st.divider()
    st.header("4. Results & Evaluation")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.info("📽️  Reference Reenactment")
        st.video(st.session_state["generated_video"])

    with res_col2:
        st.success("🎤  Patient Practice Recording")
        tab_up, tab_rec = st.tabs(["📤  Upload", "🔴  Record (Webcam)"])

        active_pat_path: "str | None" = None

        with tab_up:
            p_file = st.file_uploader(
                "Upload your practice video", type=["mp4", "mov", "webm"]
            )
            if p_file:
                if st.session_state["pat_upload_name"] != p_file.name:
                    st.session_state["pat_upload_name"] = p_file.name
                    st.session_state["pat_upload_path"] = _save(
                        p_file, save_dir="temp_patients"
                    )
                active_pat_path = st.session_state["pat_upload_path"]

        with tab_rec:
            recorded = _render_recording_panel()
            if recorded:
                active_pat_path = recorded

        if active_pat_path and os.path.isfile(active_pat_path):

            if st.session_state["_active_pat_path"] != active_pat_path:
                st.session_state["_active_pat_path"] = active_pat_path
                st.session_state["pat_is_cropped"]   = False
                st.session_state["pat_final_path"]   = active_pat_path

            st.divider()

            if not st.session_state["pat_is_cropped"]:
                st.warning("Crop your recording to the face area for best accuracy.")
                pf = extract_frame(st.session_state["pat_final_path"], second=0)

                if pf and _CROPPER_OK:
                    cp_box = st_cropper(
                        pf,
                        realtime_update = True,
                        box_color       = "#FFFFFF",
                        aspect_ratio    = (1, 1),
                        return_type     = "box",
                        key             = "pat_crop",
                    )
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button("✂️  Apply Crop", use_container_width=True):
                            with st.spinner("Cropping and preserving audio…"):
                                original_pat_path = st.session_state["pat_final_path"]

                                new_p = crop_video(
                                    original_pat_path,
                                    cp_box["left"], cp_box["top"],
                                    cp_box["width"], cp_box["height"],
                                )

                                temp_audio = original_pat_path.replace(".mp4", "_temp_audio.wav")
                                if extract_audio(original_pat_path, temp_audio):
                                    merged_p = new_p.replace(".mp4", "_with_audio.mp4")
                                    if merge_audio_video(new_p, temp_audio, merged_p):
                                        new_p = merged_p
                                    try:
                                        os.remove(temp_audio)
                                    except OSError:
                                        pass

                                _sess().add(new_p)
                                st.session_state["pat_final_path"] = new_p
                                st.session_state["pat_is_cropped"] = True
                                st.rerun()
                    with btn_c2:
                        if st.button("⏭  Skip Crop", use_container_width=True):
                            st.session_state["pat_is_cropped"] = True
                            st.rerun()
                else:
                    st.session_state["pat_is_cropped"] = True
                    st.rerun()

            else:
                st.video(st.session_state["pat_final_path"])

                ev1, ev2 = st.columns(2)
                with ev1:
                    if st.button(
                        "📊  Evaluate My Performance",
                        type                = "primary",
                        use_container_width = True,
                    ):
                        # ── GUARDRAILS ────────────────────────────────────
                        with st.spinner("Validating videos…"):
                            ref_errors = _validate_video_for_evaluation(
                                st.session_state["generated_video"],
                                label="Reference video",
                            )
                            pat_errors = _validate_video_for_evaluation(
                                st.session_state["pat_final_path"],
                                label="Patient video",
                            )

                        all_errors = ref_errors + pat_errors
                        if all_errors:
                            st.error(
                                "⚠️  Cannot run evaluation — fix the following first:"
                            )
                            for err in all_errors:
                                st.markdown(f"- {err}")
                            return

                        # ── ARTICULATION (LivePortrait expr coeffs) ───────
                        with st.spinner("Analysing articulation…"):
                            res = evaluator.evaluate(
                                reference_path = st.session_state["generated_video"],
                                patient_path   = st.session_state["pat_final_path"],
                            )

                        if "error" in res:
                            st.error(res["error"])
                            return

                        a_score = int(res.get("articulation_score", 0))
                        i_score = res.get("identity_score", 0)

                        # ── DIALOGUE ACCURACY 1 — Acoustic (ContentVec) ───
                        with st.spinner("Analysing acoustic dialogue similarity…"):
                            d_acoustic_raw = _compute_dialogue_accuracy_acoustic(
                                st.session_state["generated_video"],
                                st.session_state["pat_final_path"],
                            )
                        d_acoustic = (
                            int(d_acoustic_raw) if d_acoustic_raw is not None else None
                        )

                        # ── DIALOGUE ACCURACY 2 — Transcript (Whisper + WER) ─
                        trans_spinner_msg = (
                            f"Transcribing audio ({whisper_lang_label}) and "
                            "analysing transcript accuracy…"
                        )
                        with st.spinner(trans_spinner_msg):
                            d_trans_raw, ref_text, pat_text, d_reason = (
                                _compute_dialogue_accuracy_transcript(
                                    st.session_state["generated_video"],
                                    st.session_state["pat_final_path"],
                                    language=whisper_lang_code,
                                )
                            )
                        d_transcript = (
                            int(d_trans_raw) if d_trans_raw is not None else None
                        )

                        # ── WEIGHTED FINAL SCORE (three components) ───────
                        final_score, weight_note = _compute_weighted_final_score(
                            a_score      = float(a_score),
                            d_acoustic   = float(d_acoustic)  if d_acoustic   is not None else None,
                            d_transcript = float(d_transcript) if d_transcript is not None else None,
                        )

                        # ── DISPLAY ───────────────────────────────────────
                        st.markdown("---")
                        st.markdown("### 📊  Evaluation Results")

                        st.metric("👄  Articulation Score", f"{a_score} / 100")
                        st.progress(a_score / 100)

                        # Dialogue accuracy 1 — Acoustic
                        if d_acoustic is not None:
                            st.metric(
                                "🎵  Dialogue Accuracy (Acoustic)",
                                f"{d_acoustic} / 100",
                            )
                            st.progress(d_acoustic / 100)
                            st.caption(
                                "ContentVec embeddings + silero-vad + FastDTW (cosine) — "
                                "measures phonetic similarity."
                            )
                        else:
                            st.metric("🎵  Dialogue Accuracy (Acoustic)", "N/A")
                            st.caption(
                                "Could not compute acoustic similarity "
                                "(audio extraction or model error)."
                            )

                        # Dialogue accuracy 2 — Transcript
                        if d_transcript is not None:
                            st.metric(
                                "🗣️  Dialogue Accuracy (Transcript)",
                                f"{d_transcript} / 100",
                            )
                            st.progress(d_transcript / 100)
                            st.caption(
                                "Whisper transcription + Word Error Rate — "
                                f"measures how close the spoken words are to the "
                                f"reference. Language: **{whisper_lang_label}**."
                            )
                        else:
                            st.metric("🗣️  Dialogue Accuracy (Transcript)", "N/A")
                            if d_reason:
                                with st.expander("Show transcript-accuracy error details"):
                                    st.code(d_reason, language="text")

                        # Transcripts + word-level diff
                        if ref_text or pat_text:
                            with st.expander("📝  Transcripts & word diff", expanded=True):
                                st.markdown(f"**Reference:** _{ref_text or '(empty)'}_")
                                st.markdown(f"**You said:** _{pat_text or '(empty)'}_")
                                if ref_text and pat_text:
                                    st.markdown("**Word-by-word match:**")
                                    st.markdown(
                                        _highlight_word_diff(ref_text, pat_text),
                                        unsafe_allow_html=True,
                                    )

                        st.markdown("---")
                        if final_score >= 70:
                            colour = "#22c55e"
                        elif final_score >= 40:
                            colour = "#f59e0b"
                        else:
                            colour = "#ef4444"

                        st.markdown(
                            f'<div style="text-align:center;">'
                            f'<span style="font-size:1.1rem;font-weight:600;">'
                            f'⭐  Weighted Final Score: </span>'
                            f'<span style="font-size:1.5rem;font-weight:800;'
                            f'color:{colour};">{final_score} / 100</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(final_score / 100)

                        st.caption(
                            f"Weights: {weight_note}  ·  "
                            f"Identity mismatch: {i_score:.1f}% (lower is better)"
                        )

                        st.markdown("### 🤖 Feedback")
                        for msg in res.get("feedback", []):
                            st.info(msg)

                with ev2:
                    if st.button("🔄  Reset Recording", use_container_width=True):
                        st.session_state["pat_is_cropped"]   = False
                        st.session_state["pat_final_path"]   = active_pat_path
                        st.session_state["_active_pat_path"] = None
                        st.rerun()

if __name__ == "__main__":
    main()
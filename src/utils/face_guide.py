"""
face_guide.py
─────────────
Real-time face alignment guide processor.

Adds a semi-transparent face-shaped overlay on webcam frames and uses
MediaPipe (or OpenCV Haar as fallback) to detect whether the user's
face is centred, the right size, and properly framed.

Returns both the annotated frame (for streaming display) and a plain-text
feedback string so calling code can surface guidance in the UI.
"""

import cv2
import numpy as np

# ── MediaPipe (preferred – more accurate) ────────────────────────────────────
try:
    import mediapipe as mp
    _MEDIAPIPE_OK = True
except ImportError:
    _MEDIAPIPE_OK = False

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
_GREEN  = (50,  210,  50)
_ORANGE = (30,  160, 255)
_RED    = (50,   50, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)


class FaceGuideProcessor:
    """
    Thread-safe processor that annotates frames with a face-alignment guide.

    Usage
    -----
    >>> processor = FaceGuideProcessor()
    >>> annotated_frame, feedback = processor.process_frame(bgr_frame)
    """

    def __init__(self):
        if _MEDIAPIPE_OK:
            self._mp_fd = mp.solutions.face_detection
            self._detector = self._mp_fd.FaceDetection(
                min_detection_confidence=0.5, model_selection=0
            )
            self._backend = "mediapipe"
        else:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._backend = "opencv"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect(self, frame):
        """
        Returns (cx, cy, fw, fh) of the most prominent face, or None.
        cx/cy = centre pixel coords; fw/fh = bounding-box size in pixels.
        """
        h, w = frame.shape[:2]

        if self._backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)
            if results.detections:
                d = results.detections[0]
                b = d.location_data.relative_bounding_box
                fx = int(b.xmin * w)
                fy = int(b.ymin * h)
                fw = int(b.width  * w)
                fh = int(b.height * h)
                return (fx + fw // 2, fy + fh // 2, fw, fh)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces):
                fx, fy, fw, fh = faces[0]
                return (fx + fw // 2, fy + fh // 2, fw, fh)

        return None

    @staticmethod
    def _draw_ellipse_guide(canvas, cx, cy, ax, ay, colour, thickness=3):
        """Draw the target oval with corner tick-marks."""
        cv2.ellipse(canvas, (cx, cy), (ax, ay), 0, 0, 360, colour, thickness, cv2.LINE_AA)
        # Cardinal tick marks
        for px, py in [(cx, cy - ay), (cx, cy + ay), (cx - ax, cy), (cx + ax, cy)]:
            cv2.circle(canvas, (px, py), 5, colour, -1, cv2.LINE_AA)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Annotate *frame* with an alignment guide and directional hints.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3) – typically from cv2.VideoCapture.

        Returns
        -------
        annotated : np.ndarray
            BGR image with overlay drawn on top (same shape as *frame*).
        feedback : str
            Human-readable guidance string (e.g. "✓ Perfect – hold still!").
        """
        if frame is None:
            return None, "No frame received"

        h, w = frame.shape[:2]

        # Guide oval geometry (centred, portrait aspect)
        cx, cy = w // 2, h // 2
        ax = int(w * 0.22)   # horizontal radius
        ay = int(h * 0.34)   # vertical radius

        # ── Detect face ───────────────────────────────────────────────────────
        face = self._detect(frame)

        # ── Determine feedback ────────────────────────────────────────────────
        if face is None:
            colour   = _RED
            feedback = "No face detected — look at the camera"
        else:
            fcx, fcy, fw, fh = face
            dx        = abs(fcx - cx)
            dy        = abs(fcy - cy)
            size_ok   = (w * 0.12 < fw < w * 0.52)
            centre_ok = (dx < w * 0.09 and dy < h * 0.10)

            if centre_ok and size_ok:
                colour   = _GREEN
                feedback = "✓ Perfect — hold still!"
            elif fw < w * 0.12:
                colour   = _ORANGE
                feedback = "Move closer to the camera"
            elif fw > w * 0.52:
                colour   = _ORANGE
                feedback = "Move further from the camera"
            else:
                hints = []
                if   fcx < cx - w * 0.09: hints.append("move right")
                elif fcx > cx + w * 0.09: hints.append("move left")
                if   fcy < cy - h * 0.10: hints.append("tilt down")
                elif fcy > cy + h * 0.10: hints.append("tilt up")
                feedback = ("Adjust: " + " & ".join(hints)) if hints else "Almost there…"
                colour   = _RED

        # ── Build annotated frame ─────────────────────────────────────────────
        result = frame.copy()

        # 1. Dim area *outside* the oval (50 % opacity)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        outside = cv2.bitwise_not(mask)
        dim     = result.copy()
        dim[outside > 0] = (dim[outside > 0].astype(np.float32) * 0.45).astype(np.uint8)
        result = dim

        # 2. Draw the oval guide
        self._draw_ellipse_guide(result, cx, cy, ax, ay, colour, thickness=3)

        # 3. Top label bar
        cv2.rectangle(result, (0, 0), (w, 34), _BLACK, -1)
        cv2.putText(result, "Face Alignment Guide", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 1, cv2.LINE_AA)

        # 4. Bottom feedback bar
        bar_y = h - 42
        cv2.rectangle(result, (0, bar_y), (w, h), _BLACK, -1)
        cv2.putText(result, feedback, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)

        # 5. Colour-coded dot (top-right corner status indicator)
        cv2.circle(result, (w - 20, 17), 10, colour, -1, cv2.LINE_AA)

        return result, feedback

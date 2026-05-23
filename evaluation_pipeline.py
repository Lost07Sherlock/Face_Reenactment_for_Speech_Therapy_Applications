import numpy as np
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw
import torch
import cv2
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.cropper import Cropper
from src.utils.io import resize_to_limit
import argparse
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
class EvaluationPipeline:
    def __init__(self, inference_cfg, crop_cfg):
        self.wrapper = LivePortraitWrapper(inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.device = self.wrapper.device

        # LivePortrait specific indices that control lip/mouth shape
        # Indices: 6, 12, 14, 17, 19, 20 cover the main lip deformations
        self.LIP_INDICES = [6, 12, 14, 17, 19, 20]

    def extract_features(self, video_path):
        """
        Extracts features and performs ROBUST normalization.
        """
        cap = cv2.VideoCapture(video_path)
        motion_seq = []
        identity_vec = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = resize_to_limit(frame_rgb, 1280, 2)

            # Detect & Crop
            crop_info = self.cropper.crop_source_image(frame_rgb, self.cropper.crop_cfg)
            if crop_info is None: continue

            img_crop = crop_info['img_crop_256x256']
            I_s = self.wrapper.prepare_source(img_crop)

            with torch.no_grad():
                # --- IDENTITY (Appearance Feature) ---
                if identity_vec is None:
                    # Extract simple identity signature from first frame
                    f_s = self.wrapper.extract_feature_3d(I_s)
                    identity_vec = f_s.mean(dim=(2, 3, 4)).cpu().numpy().flatten()

                # --- MOTION (Expression Coefficients) ---
                kp_info = self.wrapper.get_kp_info(I_s)
                exp = kp_info['exp'] # [B, 21, 3]

                # Filter for Lip Articulation only
                lip_exp = exp[:, self.LIP_INDICES, :]
                motion_seq.append(lip_exp.detach().cpu().numpy().flatten())

        cap.release()

        if len(motion_seq) == 0:
            return np.array([]), None

        # --- NORMALIZATION LOGIC ---
        motion_seq = np.array(motion_seq)

        # 1. Zero-Centering (Crucial): Removes the "Resting Face" bias.
        # This ensures we compare MOVEMENT relative to neutral, not absolute face shape.
        motion_seq = motion_seq - np.mean(motion_seq, axis=0)

        # 2. Global Scaling: Normalize by the maximum movement intensity in the whole video.
        # This keeps the values roughly between -1 and 1, but preserves the relative
        # difference between "quiet" frames and "loud" frames.
        # We avoid per-frame normalization because it amplifies noise in neutral frames.
        max_val = np.max(np.abs(motion_seq))
        if max_val > 1e-5:
            motion_seq = motion_seq / max_val

        return motion_seq, identity_vec

    def evaluate(self, reference_path, patient_path):
        ref_motion, ref_id = self.extract_features(reference_path)
        pat_motion, pat_id = self.extract_features(patient_path)

        if len(ref_motion) < 5 or len(pat_motion) < 5:
            return {"error": "Video too short or face not detected."}

        # 1. Articulation Score (DTW with Euclidean)
        # Euclidean distance works better here because we preserved magnitude info.
        # Neutral frames will now be close to [0,0,...], so distance between two neutral videos will be near 0.
        distance, path = fastdtw(ref_motion, pat_motion, dist=cosine)

        # Normalize by path length
        avg_dist = distance / len(path)

        # Scoring Heuristic:
        # 0.0 error = 100%
        # 0.5 error = 0% (This is a tuned threshold for normalized lip vectors)
        articulation_score = max(0, 100 - (avg_dist * 100 / 2))

        # 2. Identity Score (Cosine Similarity)
        identity_match = 0
        if ref_id is not None and pat_id is not None:
            id_dist = cosine(ref_id, pat_id)
            identity_match = (1 - id_dist) * 100

        # 3. Feedback Logic
        feedback = []

        # Pace
        if len(pat_motion) > len(ref_motion) * 1.3:
            feedback.append("⚠️ Too Slow: Speak more naturally.")
        elif len(pat_motion) < len(ref_motion) * 0.8:
            feedback.append("⚠️ Too Fast: Slow down to articulate clearly.")

        if articulation_score > 85:
            feedback.insert(0, "✅ Excellent Articulation!")
        elif articulation_score > 60:
            feedback.insert(0, "ℹ️ Good effort. Watch the lip shape.")
        else:
            feedback.insert(0, "❌ Lip shape mismatch. Try again.")

        return {
            "articulation_score": int(articulation_score),
            "identity_score": int(identity_match),
            "feedback": feedback
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare lip articulation between a reference and a patient video "
                    "using LivePortrait's expression coefficients."
    )
    parser.add_argument(
        "reference", type=str,
        help="Path to the reference (therapist / target) video.",
    )
    parser.add_argument(
        "patient", type=str,
        help="Path to the patient (attempt) video.",
    )
    args = parser.parse_args()

    inference_cfg = InferenceConfig()
    crop_cfg = CropConfig()

    pipeline = EvaluationPipeline(inference_cfg, crop_cfg)
    result = pipeline.evaluate(args.reference, args.patient)

    print("\n" + "=" * 50)
    print("  Evaluation Result")
    print("=" * 50)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"  Articulation Score : {result['articulation_score']:3d} / 100")
    print(f"  Identity Score     : {result['identity_score']:3d} / 100")
    print("  Feedback:")
    for fb in result["feedback"]:
        print(f"    {fb}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
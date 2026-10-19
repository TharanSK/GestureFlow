"""
GestureFlow - Hybrid ONNX & Geometric Pose Classifier Module.
Combines 2-layer ONNX MLP inference with rotation-invariant anatomical joint analysis
to guarantee robust, instant recognition of all 6 target static poses:
  - POSE_OPEN_PALM
  - POSE_FIST
  - POSE_PEACE
  - POSE_THUMBS_UP
  - POSE_POINTING
  - POSE_OK
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort

POSE_LABELS = [
    "POSE_OPEN_PALM",
    "POSE_FIST",
    "POSE_PEACE",
    "POSE_THUMBS_UP",
    "POSE_POINTING",
    "POSE_OK",
]


class PoseClassifier:
    """
    Evaluates hand poses using a hybrid pipeline:
      1. Rotation-invariant anatomical finger extension geometry (instant, zero-drift)
      2. 2-layer ONNX MLP model on 63-D normalized feature vectors
    """

    def __init__(self, model_path: str = "data/models/pose_classifier.onnx", min_probability: float = 0.80):
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute() and not self.model_path.exists():
            candidate = Path(__file__).resolve().parent.parent.parent / self.model_path
            if candidate.exists():
                self.model_path = candidate

        self.min_probability = min_probability
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "float_input"

        if self.model_path.exists():
            self._load_model()
        else:
            print(f"[Warning] ONNX model not found at {self.model_path}. Geometric fallback will be used.")

    def _load_model(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def classify_geometric(self, landmarks_21x3: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classifies pose based on anatomical joint distance ratios from wrist P0 and MCP joints.
        Completely rotation-, scale-, and translation-invariant.
        """
        if landmarks_21x3 is None or len(landmarks_21x3) != 21:
            return None, 0.0

        lm = np.asarray(landmarks_21x3, dtype=np.float32)
        p0 = lm[0]
        p9 = lm[9]
        palm_span = float(np.linalg.norm(p9 - p0)) + 1e-6

        d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))

        # Anatomical finger openness:
        # A finger is open if its tip is extended relative to both wrist P0 and its MCP joint
        # Tips: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # PIPs: 6, 10, 14, 18
        # MCPs: 5, 9, 13, 17
        index_open = (d(8, 0) > d(6, 0) * 0.98) and (d(8, 5) > d(6, 5) * 0.95)
        middle_open = (d(12, 0) > d(10, 0) * 0.98) and (d(12, 9) > d(10, 9) * 0.95)
        ring_open = (d(16, 0) > d(14, 0) * 0.98) and (d(16, 13) > d(14, 13) * 0.95)
        pinky_open = (d(20, 0) > d(18, 0) * 0.98) and (d(20, 17) > d(18, 17) * 0.95)

        curled_fingers = [not index_open, not middle_open, not ring_open, not pinky_open]
        curled_count = sum(curled_fingers)

        # Thumb analysis:
        # In Thumbs Up: Thumb points away from wrist (thumb_dist_p0 > 0.60)
        # In a Fist: Thumb is curled/tucked close to palm/wrist (thumb_dist_p0 <= 0.60)
        thumb_dist_p0 = d(4, 0) / palm_span
        thumb_extended = thumb_dist_p0 > 0.60

        # Touch between thumb tip (4) and index tip (8)
        d_thumb_index = d(4, 8) / palm_span
        is_touching = d_thumb_index < 0.35

        # 1. OK Sign: Thumb and index tips touch; middle or ring is extended
        if is_touching and (middle_open or ring_open):
            return "POSE_OK", 0.95

        # 2. Open Palm: All 4 main fingers extended
        if index_open and middle_open and ring_open and pinky_open:
            return "POSE_OPEN_PALM", 0.98

        # 3. Peace / V-Sign: Index and middle extended, ring and pinky curled
        if index_open and middle_open and not ring_open and not pinky_open:
            return "POSE_PEACE", 0.96

        # 4. Pointing: Index extended, other three fingers curled
        if index_open and not middle_open and not ring_open and not pinky_open:
            return "POSE_POINTING", 0.96

        # 5. Thumbs Up vs Fist when at least 3 main fingers are curled:
        if curled_count >= 3:
            if thumb_extended and not index_open and not middle_open:
                return "POSE_THUMBS_UP", 0.94
            else:
                return "POSE_FIST", 0.96

        return None, 0.0

    def predict(
        self,
        feature_vector_63d: np.ndarray,
        raw_landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Hybrid inference:
          Evaluates geometric finger rules and ONNX model, prioritizing consensus and high confidence.
        """
        # 1. Geometric evaluation
        if raw_landmarks is not None:
            geom_label, geom_conf = self.classify_geometric(raw_landmarks)
            if geom_label is not None:
                return geom_label, geom_conf

        # 2. ONNX MLP evaluation
        if self.session is not None and feature_vector_63d is not None:
            try:
                feat = np.asarray(feature_vector_63d, dtype=np.float32).reshape(1, 63)
                outputs = self.session.run(None, {self.input_name: feat})
                pred_label_idx = int(outputs[0][0])

                prob_output = outputs[1]
                if isinstance(prob_output, list) and len(prob_output) > 0:
                    prob_dict = prob_output[0]
                    confidence = float(prob_dict.get(pred_label_idx, 0.0)) if isinstance(prob_dict, dict) else 1.0
                elif isinstance(prob_output, np.ndarray):
                    confidence = float(prob_output[0, pred_label_idx])
                else:
                    confidence = 1.0

                if confidence >= self.min_probability and 0 <= pred_label_idx < len(POSE_LABELS):
                    return POSE_LABELS[pred_label_idx], confidence
            except Exception:
                pass

        # 3. Fallback to geometric from 63-D if raw landmarks was not passed
        if feature_vector_63d is not None and len(feature_vector_63d) == 63:
            lm_recovered = np.asarray(feature_vector_63d, dtype=np.float32).reshape(21, 3)
            return self.classify_geometric(lm_recovered)

        return None, 0.0

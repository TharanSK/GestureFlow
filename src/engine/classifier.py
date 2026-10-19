"""
GestureFlow - ONNX Pose Classifier Module.
Evaluates 63-D wrist-normalized feature vectors using ONNX Runtime with confidence prob >= 0.85.
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
    Loads and runs the 2-layer ONNX MLP pose classifier.
    Rejects any prediction with confidence probability below min_probability (default 0.85).
    """

    def __init__(self, model_path: str = "data/models/pose_classifier.onnx", min_probability: float = 0.85):
        self.model_path = Path(model_path)
        self.min_probability = min_probability
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "float_input"

        if self.model_path.exists():
            self._load_model()
        else:
            print(f"[Warning] ONNX model not found at {self.model_path}. Static poses will return None.")

    def _load_model(self):
        # Configure session options for ultra-low latency CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, feature_vector_63d: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Runs inference on 63-D vector.
        Returns:
            (predicted_pose_label, confidence_probability)
            If confidence < min_probability, label is returned as None.
        """
        if self.session is None:
            return None, 0.0

        feat = np.asarray(feature_vector_63d, dtype=np.float32).reshape(1, 63)
        outputs = self.session.run(None, {self.input_name: feat})

        # Parse outputs: output_label (int), output_probability (dict)
        pred_label_idx = int(outputs[0][0])

        prob_output = outputs[1]
        if isinstance(prob_output, list) and len(prob_output) > 0:
            prob_dict = prob_output[0]
            if isinstance(prob_dict, dict):
                confidence = float(prob_dict.get(pred_label_idx, 0.0))
            elif isinstance(prob_dict, (np.ndarray, list)):
                confidence = float(prob_dict[pred_label_idx])
            else:
                confidence = 1.0
        elif isinstance(prob_output, np.ndarray):
            confidence = float(prob_output[0, pred_label_idx])
        else:
            confidence = 1.0

        if confidence >= self.min_probability and 0 <= pred_label_idx < len(POSE_LABELS):
            return POSE_LABELS[pred_label_idx], confidence

        return None, confidence

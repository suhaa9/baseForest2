from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import uuid

import numpy as np
from joblib import load
from PIL import Image

try:
    from .dataset import DEFAULT_IMAGE_SIZE, preprocess_image
except ImportError:  # script execution fallback
    from dataset import DEFAULT_IMAGE_SIZE, preprocess_image

DISPLAY_LABELS: Dict[str, str] = {
    'deforested': 'deforested',
    'non-deforested': 'aforested',
}


class Predictor:
    def __init__(self, model_path: str):
        bundle = load(model_path)
        try:
            self.model = bundle['model']
            self.classes = list(bundle['classes'])
            self.scaler = bundle['scaler']
            self.image_size = tuple(bundle.get('image_size', DEFAULT_IMAGE_SIZE))
            self.metrics = bundle.get('metrics', {})
        except KeyError as exc:
            raise RuntimeError('Loaded model bundle is missing required keys. Retrain the model with the latest training script.') from exc

        trace_dir = os.environ.get('PREDICT_TRACE_DIR')
        if trace_dir:
            self.trace_dir = Path(trace_dir).expanduser().resolve()
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.trace_dir = None

    def _display_label(self, label: str) -> str:
        return DISPLAY_LABELS.get(label, label)

    def _prepare_vectors(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        raw_vec = preprocess_image(img, image_size=self.image_size, flatten=True)
        scaled_vec = self.scaler.transform([raw_vec])[0]
        return raw_vec, scaled_vec

    def _maybe_trace(self, raw_vec: np.ndarray, scaled_vec: np.ndarray, payload: Dict[str, Any]) -> str | None:
        if not self.trace_dir:
            return None

        trace_payload = {
            'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
            **payload,
        }
        trace_id = trace_payload.get('trace_id') or uuid.uuid4().hex
        trace_payload['trace_id'] = trace_id

        trace_path = self.trace_dir / f'{trace_id}.npz'
        np.savez_compressed(
            trace_path,
            raw=raw_vec.astype(np.float32),
            scaled=scaled_vec.astype(np.float32),
            metadata=json.dumps(trace_payload),
        )
        return str(trace_path)

    def predict(
        self,
        img: Image.Image,
        top_k: int = 3,
        *,
        capture_vectors: bool = False,
        trace_context: Dict[str, Any] | None = None,
    ):
        raw_vec, scaled_vec = self._prepare_vectors(img)
        probabilities = self.model.predict_proba([scaled_vec])[0]
        top_k = min(top_k, len(self.classes))
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        best_idx = int(top_indices[0])
        top_scores = [
            {
                'class': self._display_label(self.classes[i]),
                'index': int(i),
                'confidence': float(probabilities[i]),
            }
            for i in top_indices
        ]
        prediction = {
            'class': self._display_label(self.classes[best_idx]),
            'index': best_idx,
            'confidence': float(probabilities[best_idx]),
            'top_scores': top_scores,
            'metrics': self.metrics,
            'heuristic': None,
        }

        trace_payload: Dict[str, Any] = {
            'prediction': {
                'class': prediction['class'],
                'confidence': prediction['confidence'],
                'index': prediction['index'],
            },
            'top_scores': top_scores,
        }
        if trace_context:
            trace_payload['context'] = trace_context

        trace_file = self._maybe_trace(raw_vec, scaled_vec, trace_payload)
        if trace_file:
            prediction['trace_file'] = trace_file

        if capture_vectors:
            return prediction, raw_vec, scaled_vec
        return prediction


class GreenPixelPredictor:
    def __init__(
        self,
        *,
        green_delta_threshold: float = 0.05,
        coverage_threshold: float = 0.45,
    ) -> None:
        self.green_delta_threshold = float(green_delta_threshold)
        self.coverage_threshold = float(coverage_threshold)

    def _compute_stats(self, img: Image.Image) -> Dict[str, Any]:
        arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]

        greenness = green - np.maximum(red, blue)
        mask = greenness >= self.green_delta_threshold

        total_pixels = mask.size
        green_pixels = int(mask.sum())
        green_ratio = green_pixels / total_pixels if total_pixels else 0.0

        return {
            'green_ratio': float(green_ratio),
            'avg_green_channel': float(green.mean()),
            'std_green_channel': float(green.std()),
            'greenness_delta_mean': float(greenness.mean()),
            'green_pixels': green_pixels,
            'total_pixels': int(total_pixels),
        }

    def predict(self, img: Image.Image, **_: Any) -> Dict[str, Any]:
        stats = self._compute_stats(img)
        predicted_class = 'aforested' if stats['green_ratio'] >= self.coverage_threshold else 'deforested'

        confidence = stats['green_ratio']
        top_scores = [
            {
                'class': 'aforested',
                'index': 1,
                'confidence': float(confidence),
            },
            {
                'class': 'deforested',
                'index': 0,
                'confidence': float(max(0.0, 1.0 - confidence)),
            },
        ]

        prediction = {
            'class': predicted_class,
            'index': 1 if predicted_class == 'aforested' else 0,
            'confidence': float(confidence),
            'top_scores': top_scores,
            'metrics': None,
            'heuristic': {
                **stats,
                'green_delta_threshold': self.green_delta_threshold,
                'coverage_threshold': self.coverage_threshold,
            },
        }

        return prediction

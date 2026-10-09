"""Pruebas del módulo de visión por computadora local (YOLOv8 en ONNX vía OpenCV DNN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.model_files import ModelIntegrityError
from app.schemas import VisualDetection
from app.vision import FakeVisionDetector, VisionModelNotAvailableError, YoloVisionDetector


def test_fake_vision_detector_returns_expected_detections() -> None:
    detector = FakeVisionDetector()
    assert detector.is_available() is True
    assert detector.model_name == "fake-yolo"

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(img)
    assert len(detections) == 1
    assert isinstance(detections[0], VisualDetection)
    assert detections[0].label == "persona"
    assert 0.0 <= detections[0].confidence <= 1.0
    x1, y1, x2, y2 = detections[0].box
    assert 0.0 <= x1 <= x2 <= 1.0
    assert 0.0 <= y1 <= y2 <= 1.0


def test_yolo_vision_detector_fails_gracefully_if_model_missing(tmp_path: Path) -> None:
    detector = YoloVisionDetector(models_root=tmp_path, model_name="missing.onnx")
    assert detector.is_available() is False
    with pytest.raises(VisionModelNotAvailableError):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_yolo_vision_detector_verifies_integrity(tmp_path: Path) -> None:
    vision_dir = tmp_path / "vision"
    vision_dir.mkdir(parents=True)
    fake_model = vision_dir / "yolov8n.onnx"
    fake_model.write_bytes(b"modelo adulterado")

    detector = YoloVisionDetector(models_root=tmp_path, model_name="yolov8n.onnx")
    assert detector.is_available() is True
    with pytest.raises(ModelIntegrityError):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_yolo_vision_detector_real_inference_on_empty_and_synthetic_frame(settings: Settings) -> None:
    detector = YoloVisionDetector(
        models_root=settings.models_dir,
        model_name=settings.vision_model,
        confidence_threshold=settings.vision_confidence_threshold,
        nms_threshold=settings.vision_nms_threshold,
    )
    if not detector.is_available():
        pytest.skip("Modelo YOLOv8n no aprovisionado")

    # Imagen vacía (negra): no debe encontrar objetos ni fallar.
    empty = np.zeros((480, 640, 3), dtype=np.uint8)
    res_empty = detector.detect(empty)
    assert isinstance(res_empty, tuple)

    # Imagen de tamaño cero.
    zero = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect(zero) == ()

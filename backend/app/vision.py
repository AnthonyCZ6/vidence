"""Detección visual local con YOLOv8 en formato ONNX (OpenCV DNN en CPU).

El modelo se ejecuta 100 % offline utilizando el motor DNN de OpenCV, sin
requerir PyTorch ni ONNXRuntime pesado. Los pesos se verifican contra
models/MANIFEST.json.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

from app.model_files import verify_model_files
from app.schemas import VisualDetection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 80 clases COCO estándar en español para la interfaz de Vidence.
COCO_CLASSES: tuple[str, ...] = (
    "persona", "bicicleta", "automóvil", "motocicleta", "avión", "autobús", "tren", "camión", "bote",
    "semáforo", "boca de incendios", "señal de alto", "parquímetro", "banca", "pájaro", "gato",
    "perro", "caballo", "oveja", "vaca", "elefante", "oso", "cebra", "jirafa", "mochila",
    "paraguas", "bolso", "corbata", "maleta", "frisbee", "esquís", "snowboard", "pelota",
    "cometa", "bate de béisbol", "guante de béisbol", "patineta", "tabla de surf", "raqueta de tenis",
    "botella", "copa de vino", "taza", "tenedor", "cuchillo", "cuchara", "tazón", "plátano",
    "manzana", "sándwich", "naranja", "brócoli", "zanahoria", "hot dog", "pizza", "dona",
    "pastel", "silla", "sofá", "planta en maceta", "cama", "mesa de comedor", "inodoro", "televisor",
    "computadora portátil", "ratón", "control remoto", "teclado", "teléfono celular", "microondas",
    "horno", "tostadora", "fregadero", "refrigerador", "libro", "reloj", "jarrón", "tijeras",
    "oso de peluche", "secador de pelo", "cepillo de dientes",
)


class VisionError(Exception):
    """Error base del módulo de visión."""


class VisionModelNotAvailableError(VisionError):
    """El modelo de visión no está aprovisionado en models/vision/."""


class VisionDetector(Protocol):
    @property
    def model_name(self) -> str: ...

    def is_available(self) -> bool: ...

    def detect(self, image_bgr: np.ndarray) -> tuple[VisualDetection, ...]: ...


class FakeVisionDetector:
    """Detector sintético para pruebas unitarias de pipeline y API."""

    def __init__(self, detections: Sequence[VisualDetection] | None = None) -> None:
        self._detections = tuple(detections) if detections is not None else (
            VisualDetection(label="persona", confidence=0.92, box=(0.1, 0.15, 0.5, 0.85)),
        )

    @property
    def model_name(self) -> str:
        return "fake-yolo"

    def is_available(self) -> bool:
        return True

    def detect(self, image_bgr: np.ndarray) -> tuple[VisualDetection, ...]:
        return self._detections


class YoloVisionDetector:
    def __init__(
        self,
        *,
        models_root: Path,
        model_name: str,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ) -> None:
        self._models_root = models_root
        self._model_name = model_name
        self._model_path = models_root / "vision" / model_name
        self._conf_thresh = confidence_threshold
        self._nms_thresh = nms_threshold
        self._net: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return self._model_path.is_file()

    def _load(self) -> Any:
        with self._lock:
            if self._net is None:
                if not self.is_available():
                    raise VisionModelNotAvailableError(
                        f"Modelo de visión '{self._model_name}' no encontrado; ejecute scripts/provision_models.py"
                    )
                verify_model_files(self._models_root, f"vision/{self._model_name}")
                logger.info("Cargando modelo de visión ONNX: %s", self._model_path.name)
                net = cv2.dnn.readNetFromONNX(str(self._model_path))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net = net
            return self._net

    def detect(self, image_bgr: np.ndarray) -> tuple[VisualDetection, ...]:
        net = self._load()
        orig_h, orig_w = image_bgr.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return ()

        # Letterbox para preservar la relación de aspecto exacta en el tensor 640x640.
        scale = min(INPUT_WIDTH / orig_w, INPUT_HEIGHT / orig_h)
        scaled_w, scaled_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        pad_x = (INPUT_WIDTH - scaled_w) // 2
        pad_y = (INPUT_HEIGHT - scaled_h) // 2
        canvas[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        with self._lock:
            net.setInput(blob)
            output = net.forward()

        # YOLOv8 produce shape: (1, 84, 8400) -> transponer a (8400, 84)
        if len(output.shape) == 3 and output.shape[1] == 84:
            predictions = output[0].T
        elif len(output.shape) == 2 and output.shape[1] == 84:
            predictions = output
        else:
            logger.warning("Formato de salida inesperado en modelo de visión: %s", output.shape)
            return ()

        boxes_for_nms: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        num_classes = len(COCO_CLASSES)
        for row in predictions:
            scores = row[4 : 4 + num_classes]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence >= self._conf_thresh:
                cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                boxes_for_nms.append([x1, y1, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes_for_nms:
            return ()

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, self._conf_thresh, self._nms_thresh)
        if len(indices) == 0:
            return ()

        selected_indices = [int(i[0] if isinstance(i, (list, tuple, np.ndarray)) else i) for i in indices]

        detections: list[VisualDetection] = []
        for idx in selected_indices:
            cid = class_ids[idx]
            label = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"clase_{cid}"
            conf = confidences[idx]
            bx, by, bw, bh = boxes_for_nms[idx]

            # Revertir el padding de letterbox y normalizar al tamaño original [0.0, 1.0].
            orig_x1 = max(0.0, min(1.0, (bx - pad_x) / scaled_w))
            orig_y1 = max(0.0, min(1.0, (by - pad_y) / scaled_h))
            orig_x2 = max(0.0, min(1.0, (bx + bw - pad_x) / scaled_w))
            orig_y2 = max(0.0, min(1.0, (by + bh - pad_y) / scaled_h))

            # Asegurar coordenadas válidas y ordenadas.
            box = (
                round(min(orig_x1, orig_x2), 4),
                round(min(orig_y1, orig_y2), 4),
                round(max(orig_x1, orig_x2), 4),
                round(max(orig_y1, orig_y2), 4),
            )
            detections.append(VisualDetection(label=label, confidence=round(conf, 4), box=box))

        # Ordenar por nivel de confianza descendente.
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return tuple(detections)

"""Muestreo de fotogramas con OpenCV: 1 fotograma cada N segundos.

Lectura secuencial con `grab()` (sin convertir cada cuadro) y marca de tiempo
real del cuadro (`CAP_PROP_POS_MSEC`): los videos de móvil suelen tener
frame rate variable y el seek por índice sería impreciso. Los JPEG se
codifican en memoria y se cifran directamente: no tocan el disco en claro.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.media import MediaError

JPEG_QUALITY = 90
_TIMESTAMP_TOLERANCE_S = 1e-3


class FrameExtractionError(MediaError):
    """No se pudo leer o codificar el video."""


@dataclass(frozen=True)
class ExtractedFrame:
    index: int
    timestamp_s: float
    width: int
    height: int
    jpeg: bytes
    image_bgr: np.ndarray | None = None


def effective_interval(requested_s: float, duration_s: float | None, max_frames: int) -> float:
    """Amplía el intervalo en videos largos para no superar `max_frames`."""
    if duration_s and duration_s > 0:
        return max(requested_s, duration_s / max_frames)
    return requested_s


def iter_frames(video_path: Path, *, interval_s: float, max_frames: int) -> Iterator[ExtractedFrame]:
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise FrameExtractionError("OpenCV no pudo abrir el video")
    try:
        next_timestamp = 0.0
        index = 0
        while index < max_frames and capture.grab():
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp + _TIMESTAMP_TOLERANCE_S < next_timestamp:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                continue
            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not encoded:
                raise FrameExtractionError("No se pudo codificar un fotograma a JPEG")
            height, width = frame.shape[:2]
            yield ExtractedFrame(index, round(timestamp, 3), width, height, buffer.tobytes(), frame)
            index += 1
            next_timestamp = (math.floor(timestamp / interval_s + _TIMESTAMP_TOLERANCE_S) + 1) * interval_s
    finally:
        capture.release()

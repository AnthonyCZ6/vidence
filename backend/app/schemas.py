"""Modelos de datos inmutables del manifiesto de evidencia.

Neutralidad por diseño: estos esquemas solo describen hechos técnicos
(metadatos, hashes, linaje de procesamiento, transcripción y fotogramas).
No existe ningún campo de riesgo, amenaza, sentimiento ni puntuación.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ORIGINAL_NAME = "original.bin"
INGEST_NAME = "ingest.json"
MANIFEST_NAME = "manifest.json"

AssetRole = Literal["original", "original_preview", "enhanced"]
EvidenceSource = Literal["upload", "recording"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StreamInfo(_Frozen):
    index: int
    codec_type: Literal["audio", "video"]
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_s: float | None = None


class MediaInfo(_Frozen):
    format_name: str
    format_long_name: str | None = None
    duration_s: float | None = None
    size_bytes: int | None = None
    bit_rate: int | None = None
    creation_time: str | None = None
    has_audio: bool
    has_video: bool
    streams: tuple[StreamInfo, ...] = ()
    tags: dict[str, str] = Field(default_factory=dict)

    @property
    def audio_duration_s(self) -> float | None:
        """Duración de la primera pista de audio; si no consta, la del contenedor."""
        for stream in self.streams:
            if stream.codec_type == "audio" and stream.duration_s:
                return stream.duration_s
        return self.duration_s


class Word(_Frozen):
    start: float
    end: float
    text: str
    probability: float


class Segment(_Frozen):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: tuple[Word, ...] = ()


class Transcript(_Frozen):
    engine: str
    model: str
    device: str
    compute_type: str
    language: str
    language_probability: float
    duration_s: float
    source: str
    segments: tuple[Segment, ...]
    notice: str


class ProcessingRecord(_Frozen):
    """Linaje reproducible de un derivado: herramienta, versión y argumentos (sin rutas locales)."""

    tool: str
    tool_version: str
    arguments: tuple[str, ...]


class StoredAsset(_Frozen):
    name: str
    role: AssetRole
    media_type: str
    sha256: str
    size_bytes: int
    derived_from_sha256: str | None = None
    processing: ProcessingRecord | None = None


class VisualDetection(_Frozen):
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # [x1, y1, x2, y2] normalizados en [0.0, 1.0]


class FrameAsset(_Frozen):
    index: int
    timestamp_s: float
    name: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    detections: tuple[VisualDetection, ...] = ()


class IngestRecord(_Frozen):
    evidence_id: str
    original_filename: str
    source: EvidenceSource
    received_at: datetime
    original: StoredAsset


class Manifest(_Frozen):
    schema_version: int = 1
    evidence_id: str
    ingest: IngestRecord
    processed_at: datetime
    integrity_verified: bool
    media: MediaInfo
    derivatives: tuple[StoredAsset, ...]
    transcript: Transcript | None
    frames: tuple[FrameAsset, ...]
    frame_interval_s: float | None
    frames_source: str | None
    visual_summary: dict[str, int] = Field(default_factory=dict)
    notices: tuple[str, ...]

    def media_type_for(self, name: str) -> str | None:
        """Tipo MIME de un objeto listado en el manifiesto; None si no pertenece a esta evidencia."""
        for asset in (self.ingest.original, *self.derivatives):
            if asset.name == name:
                return asset.media_type
        if any(frame.name == name for frame in self.frames):
            return "image/jpeg"
        return None

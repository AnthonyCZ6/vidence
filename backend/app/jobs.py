"""Cola de trabajos en memoria con un único worker.

Un solo hilo procesa las evidencias en orden de llegada: Whisper y ffmpeg
saturan la CPU y con hardware limitado no conviene paralelizar. CTranslate2
libera el GIL durante la inferencia, así que la API sigue respondiendo.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.crypto import CryptoError
from app.media import FfmpegError, UnsupportedMediaError
from app.model_files import ModelIntegrityError
from app.pipeline import OriginalIntegrityError, ProgressReporter, Stage
from app.transcribe import TranscriptionError

logger = logging.getLogger(__name__)

STAGE_MESSAGES: Mapping[Stage, str] = {
    Stage.QUEUED: "En cola",
    Stage.VERIFYING: "Verificando la integridad del original (SHA-256)",
    Stage.PROBING: "Leyendo metadatos con ffprobe",
    Stage.ENHANCING: "Mejorando audio y video (RNNoise, normalización, nitidez)",
    Stage.TRANSCRIBING: "Transcribiendo con Whisper (CPU int8)",
    Stage.FRAMES: "Analizando video y extrayendo fotogramas con IA (YOLOv8)",
    Stage.FINALIZING: "Cifrando y guardando resultados",
    Stage.DONE: "Completado",
    Stage.FAILED: "Error",
}

PipelineRunner = Callable[[str, ProgressReporter], object]


class JobStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    evidence_id: str
    stage: Stage
    progress: float | None
    message: str
    error: str | None = None
    updated_at: datetime
    # Etapas del pipeline realmente ejecutadas, en orden. La interfaz consulta cada pocos
    # cientos de ms y no debe deducir "omitida" de una etapa breve que no alcanzó a ver.
    stages_seen: tuple[Stage, ...] = ()


_TERMINAL_STAGES = frozenset({Stage.QUEUED, Stage.DONE, Stage.FAILED})


def describe_failure(exc: BaseException) -> str:
    """Mensaje apto para la interfaz: sin rutas, trazas ni contenido de la evidencia."""
    match exc:
        case UnsupportedMediaError():
            return f"Formato no compatible: {exc}"
        case FfmpegError():
            return "ffmpeg no pudo procesar el archivo."
        case OriginalIntegrityError() | CryptoError():
            return "Falló la verificación de integridad del original cifrado."
        case ModelIntegrityError() | TranscriptionError():
            return f"Modelo de IA no disponible: {exc}"
        case _:
            return "Error interno al procesar la evidencia; consulte el registro local."


class JobManager:
    def __init__(self, runner: PipelineRunner) -> None:
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vidence-worker")
        self._jobs: dict[str, JobStatus] = {}
        self._latest_by_evidence: dict[str, str] = {}
        self._lock = threading.Lock()

    def submit(self, evidence_id: str) -> JobStatus:
        job_id = uuid.uuid4().hex
        status = JobStatus(
            job_id=job_id,
            evidence_id=evidence_id,
            stage=Stage.QUEUED,
            progress=None,
            message=STAGE_MESSAGES[Stage.QUEUED],
            updated_at=datetime.now(UTC),
        )
        with self._lock:
            self._jobs[job_id] = status
            self._latest_by_evidence[evidence_id] = job_id
        self._executor.submit(self._execute, job_id, evidence_id)
        return status

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest_for_evidence(self, evidence_id: str) -> JobStatus | None:
        with self._lock:
            job_id = self._latest_by_evidence.get(evidence_id)
            return self._jobs.get(job_id) if job_id else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, job_id: str, evidence_id: str) -> None:
        try:
            self._runner(evidence_id, lambda stage, progress: self._update(job_id, stage, progress))
        except Exception as exc:
            logger.exception("Falló el procesamiento del trabajo %s (evidencia %s)", job_id, evidence_id)
            self._update(job_id, Stage.FAILED, None, error=describe_failure(exc))
        else:
            self._update(job_id, Stage.DONE, 1.0)

    def _update(self, job_id: str, stage: Stage, progress: float | None, error: str | None = None) -> None:
        with self._lock:
            current = self._jobs[job_id]
            seen = current.stages_seen
            if stage not in _TERMINAL_STAGES and stage not in seen:
                seen = (*seen, stage)
            self._jobs[job_id] = current.model_copy(
                update={
                    "stage": stage,
                    "progress": progress,
                    "message": STAGE_MESSAGES[stage],
                    "error": error,
                    "updated_at": datetime.now(UTC),
                    "stages_seen": seen,
                }
            )

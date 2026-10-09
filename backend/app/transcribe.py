"""Transcripción local con faster-whisper en CPU (cuantización int8).

El modelo se carga desde una carpeta local verificada contra MANIFEST.json,
con `local_files_only=True`: nunca se consulta Hugging Face. El VAD (Silero)
viene incluido en el paquete faster-whisper.

Whisper recibe muestras ya decodificadas (numpy), no una ruta: ffmpeg, con sus
listas blancas de protocolo y formato, es el único decodificador del sistema.
"""

from __future__ import annotations

import logging
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app.media import ASR_SAMPLE_RATE
from app.model_files import verify_model_files
from app.schemas import Segment, Transcript, Word

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
VAD_MIN_SILENCE_MS = 500
_PCM16_SAMPLE_WIDTH = 2
_PCM16_FULL_SCALE = 32768.0
TRANSCRIPT_NOTICE = (
    "Transcripción automática generada por IA local (faster-whisper). Puede contener errores u "
    "omisiones; la evidencia primaria es siempre el audio original."
)

ProgressCallback = Callable[[float], None]


class TranscriptionError(Exception):
    """Error base de transcripción."""


class ModelNotAvailableError(TranscriptionError):
    """El modelo Whisper no está aprovisionado en models/."""


class Transcriber(Protocol):
    @property
    def model_name(self) -> str: ...

    def is_available(self) -> bool: ...

    def transcribe(self, audio_path: Path, *, source_label: str, on_progress: ProgressCallback) -> Transcript: ...


def load_pcm16_mono(path: Path) -> np.ndarray:
    """Lee el WAV producido por `media.extract_asr_audio` como float32 en [-1, 1)."""
    with wave.open(str(path), "rb") as wav:
        layout = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        if layout != (1, _PCM16_SAMPLE_WIDTH, ASR_SAMPLE_RATE):
            raise TranscriptionError(f"El audio para ASR debe ser PCM 16 bits mono {ASR_SAMPLE_RATE} Hz")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / _PCM16_FULL_SCALE


def to_segment(raw: Any) -> Segment:
    return Segment(
        id=int(raw.id),
        start=round(float(raw.start), 3),
        end=round(float(raw.end), 3),
        text=str(raw.text).strip(),
        avg_logprob=round(float(raw.avg_logprob), 4),
        no_speech_prob=round(float(raw.no_speech_prob), 4),
        words=tuple(
            Word(
                start=round(float(w.start), 3),
                end=round(float(w.end), 3),
                text=str(w.word),
                probability=round(float(w.probability), 4),
            )
            for w in (raw.words or ())
        ),
    )


class WhisperTranscriber:
    def __init__(
        self, *, models_root: Path, model_name: str, language: str | None, cpu_threads: int, beam_size: int
    ) -> None:
        self._models_root = models_root
        self._model_name = model_name
        self._model_dir = models_root / "whisper" / model_name
        self._language = language
        self._cpu_threads = cpu_threads
        self._beam_size = beam_size
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return (self._model_dir / "model.bin").is_file()

    def _load(self) -> WhisperModel:
        with self._lock:
            if self._model is None:
                if not self.is_available():
                    raise ModelNotAvailableError(
                        f"Modelo Whisper '{self._model_name}' no encontrado; ejecute scripts/provision_models.py"
                    )
                verified = verify_model_files(self._models_root, f"whisper/{self._model_name}/")
                logger.info("Modelo Whisper '%s' verificado (%d archivos); cargando", self._model_name, verified)
                from faster_whisper import WhisperModel  # import diferido: carga pesada

                self._model = WhisperModel(
                    str(self._model_dir),
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=self._cpu_threads,
                    local_files_only=True,
                )
            return self._model

    def transcribe(self, audio_path: Path, *, source_label: str, on_progress: ProgressCallback) -> Transcript:
        """`audio_path` debe ser PCM 16 bits mono 16 kHz (ver `media.extract_asr_audio`)."""
        model = self._load()
        segments_iter, info = model.transcribe(
            load_pcm16_mono(audio_path),
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        segments: list[Segment] = []
        for raw in segments_iter:
            segments.append(to_segment(raw))
            if info.duration > 0:
                on_progress(min(raw.end / info.duration, 1.0))
        return Transcript(
            engine="faster-whisper",
            model=self._model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=info.language,
            language_probability=round(float(info.language_probability), 4),
            duration_s=round(float(info.duration), 3),
            source=source_label,
            segments=tuple(segments),
            notice=TRANSCRIPT_NOTICE,
        )

"""ffprobe/ffmpeg locales: validación de formato, mejora de audio/video y extracción para ASR.

Seguridad: ffmpeg se invoca sin shell, con argumentos en lista, y siempre con
`-protocol_whitelist file` y `-format_whitelist` antes de la entrada. Así un
archivo malicioso (p. ej. una playlist HLS disfrazada de .mp4) no puede hacer
que ffmpeg lea URLs ni otros archivos locales.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any

from app.schemas import MediaInfo, StreamInfo

ALLOWED_DEMUXERS = ("mov", "matroska", "mp3", "wav", "ogg", "flac", "aac")
INPUT_GUARD = ("-protocol_whitelist", "file", "-format_whitelist", ",".join(ALLOWED_DEMUXERS))

# Tipo MIME por extensión aceptada en la subida (no se confía en el Content-Type del cliente).
UPLOAD_MEDIA_TYPES: Mapping[str, str] = {
    ".mp4": "video/mp4",
    ".m4a": "audio/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}

HIGHPASS = "highpass=f=80"
FFT_DENOISE = "afftdn=nr=12:nf=-30:tn=1"
# arnndn (ffmpeg 9.0.2) procesa la última trama incompleta con memoria sin inicializar:
# de forma no determinista emite NaN o picos de hasta +290 dB. Se añade silencio al final
# para que esa trama caiga en el relleno y luego se recorta a la duración original. Como
# segunda defensa, NaN/Inf → 0 y muestras limitadas a ±1 antes de loudnorm (cuyo estado
# quedaría contaminado y cuya medición saldría imposible, p. ej. +29.95 LUFS).
RNNOISE_TAIL_PAD = "apad=pad_dur=0.1"
FINITE_SAMPLES = "aeval=exprs='if(isnan(val(ch))+isinf(val(ch)),0,clip(val(ch),-1,1))':c=same"
LOUDNORM_TARGET = "I=-16:TP=-1.5:LRA=11"
OUTPUT_SAMPLE_RATE = 48000
ASR_SAMPLE_RATE = 16000
EVEN_DIMENSIONS = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
PREVIEW_VIDEO_FILTER = f"{EVEN_DIMENSIONS},format=yuv420p"
ENHANCE_VIDEO_FILTER = f"hqdn3d=1.5:1.5:6:6,unsharp=5:5:0.8:5:5:0.0,{EVEN_DIMENSIONS},format=yuv420p"
VIDEO_CODEC_ARGS = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart")
AAC_ARGS = ("-c:a", "aac", "-b:a", "192k")
FLAC_ARGS = ("-c:a", "flac")

_SAFE_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LOUDNORM_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)
_LOUDNORM_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
# Rangos que acepta loudnorm para measured_I, measured_TP, measured_LRA, measured_thresh y offset.
_LOUDNORM_RANGES = ((-99.0, 0.0), (-99.0, 99.0), (0.0, 99.0), (-99.0, 0.0), (-99.0, 99.0))
_STDERR_TAIL_CHARS = 2000
_MAX_TAGS = 64
_MAX_TAG_VALUE_CHARS = 512
# Una caída nativa (p. ej. 0xC0000005) se reintenta una vez: cada llamada es
# determinista y sobrescribe su salida (-y). Los errores normales no se reintentan.
CRASH_RETRIES = 1
# NTSTATUS de error (0xC...): violación de acceso, desbordamiento de pila, etc. Los
# errores propios de ffmpeg (AVERROR negativos) llegan como 0xFFFFFFxx y no son caídas.
_NTSTATUS_ERROR_FIRST = 0xC0000000
_NTSTATUS_ERROR_LAST = 0xCFFFFFFF

logger = logging.getLogger(__name__)


class MediaError(Exception):
    """Error base de procesamiento multimedia."""


class UnsupportedMediaError(MediaError):
    """El archivo no es un audio/video en un contenedor permitido."""


class FfmpegError(MediaError):
    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


@dataclass(frozen=True)
class FfmpegTools:
    ffmpeg: str
    ffprobe: str
    timeout_s: int
    version: str

    @classmethod
    def resolve(cls, ffmpeg_bin: str, ffprobe_bin: str, timeout_s: int) -> FfmpegTools:
        ffmpeg, ffprobe = shutil.which(ffmpeg_bin), shutil.which(ffprobe_bin)
        if ffmpeg is None or ffprobe is None:
            raise MediaError(
                "No se encontró ffmpeg/ffprobe. Instálelo o configure VIDENCE_FFMPEG_BIN y VIDENCE_FFPROBE_BIN."
            )
        first_line = _run([ffmpeg, "-hide_banner", "-version"], timeout_s=timeout_s).stdout.splitlines()[0]
        return cls(ffmpeg, ffprobe, timeout_s, first_line.split(" Copyright")[0])


@dataclass(frozen=True)
class _ProcessOutput:
    stdout: str
    stderr: str


@dataclass(frozen=True)
class LoudnormMeasurement:
    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


@dataclass(frozen=True)
class RenderResult:
    output: Path
    arguments: tuple[str, ...]


def is_crash_exit(returncode: int) -> bool:
    """Terminación anómala: señal en POSIX (negativo) o NTSTATUS de error en Windows (0xC0000000-0xCFFFFFFF)."""
    return returncode < 0 or _NTSTATUS_ERROR_FIRST <= returncode <= _NTSTATUS_ERROR_LAST


def _invoke(args: Sequence[str], timeout_s: int, cwd: Path | None) -> subprocess.CompletedProcess[bytes]:
    tool = Path(args[0]).stem
    try:
        return subprocess.run(  # noqa: S603  (lista de argumentos, sin shell)
            list(args), cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"{tool} superó el tiempo límite de {timeout_s} s") from exc
    except OSError as exc:
        raise FfmpegError(f"No se pudo ejecutar {tool}", str(exc)) from exc


def _run(args: Sequence[str], *, timeout_s: int, cwd: Path | None = None) -> _ProcessOutput:
    tool = Path(args[0]).stem
    completed = _invoke(args, timeout_s, cwd)
    for _ in range(CRASH_RETRIES):
        if not is_crash_exit(completed.returncode):
            break
        logger.warning("%s terminó de forma anómala (código %#x); se reintenta", tool, completed.returncode)
        completed = _invoke(args, timeout_s, cwd)
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        tail = stderr[-_STDERR_TAIL_CHARS:]
        logger.warning("%s terminó con código %#x: %s", tool, completed.returncode, tail.strip() or "(sin stderr)")
        raise FfmpegError(f"{tool} terminó con código {completed.returncode}", tail)
    return _ProcessOutput(stdout, stderr)


# --- Análisis con ffprobe ----------------------------------------------------------


def probe(tools: FfmpegTools, path: Path) -> MediaInfo:
    args = [
        tools.ffprobe,
        "-v",
        "error",
        *INPUT_GUARD,
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        output = _run(args, timeout_s=tools.timeout_s)
        data = json.loads(output.stdout)
    except FfmpegError as exc:
        raise UnsupportedMediaError("Formato no reconocido o no permitido") from exc
    except json.JSONDecodeError as exc:
        raise UnsupportedMediaError("ffprobe devolvió datos ilegibles") from exc
    return parse_probe(data)


def parse_probe(data: Mapping[str, Any]) -> MediaInfo:
    fmt: Mapping[str, Any] = data.get("format") or {}
    format_name = str(fmt.get("format_name", ""))
    if not set(format_name.split(",")) & set(ALLOWED_DEMUXERS):
        raise UnsupportedMediaError(f"Contenedor no permitido: {format_name or 'desconocido'}")
    streams = tuple(
        _parse_stream(raw)
        for raw in data.get("streams") or ()
        if raw.get("codec_type") in {"audio", "video"} and not _is_cover_art(raw)
    )
    has_audio = any(s.codec_type == "audio" for s in streams)
    has_video = any(s.codec_type == "video" for s in streams)
    if not (has_audio or has_video):
        raise UnsupportedMediaError("El archivo no contiene pistas de audio ni de video")
    tags = _clean_tags(fmt.get("tags"))
    return MediaInfo(
        format_name=format_name,
        format_long_name=fmt.get("format_long_name"),
        duration_s=_to_float(fmt.get("duration")),
        size_bytes=_to_int(fmt.get("size")),
        bit_rate=_to_int(fmt.get("bit_rate")),
        creation_time=tags.get("creation_time") or _first_stream_tag(data, "creation_time"),
        has_audio=has_audio,
        has_video=has_video,
        streams=streams,
        tags=tags,
    )


def _is_cover_art(raw: Mapping[str, Any]) -> bool:
    return bool((raw.get("disposition") or {}).get("attached_pic"))


def _parse_stream(raw: Mapping[str, Any]) -> StreamInfo:
    return StreamInfo(
        index=int(raw.get("index", 0)),
        codec_type=raw["codec_type"],
        codec_name=raw.get("codec_name"),
        width=_to_int(raw.get("width")),
        height=_to_int(raw.get("height")),
        frame_rate=_parse_rate(raw.get("avg_frame_rate")),
        sample_rate=_to_int(raw.get("sample_rate")),
        channels=_to_int(raw.get("channels")),
        duration_s=_to_float(raw.get("duration")),
    )


def _clean_tags(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    items = list(raw.items())[:_MAX_TAGS]
    return {str(key)[:64]: str(value)[:_MAX_TAG_VALUE_CHARS] for key, value in items}


def _first_stream_tag(data: Mapping[str, Any], key: str) -> str | None:
    for raw in data.get("streams") or ():
        value = (raw.get("tags") or {}).get(key)
        if value:
            return str(value)
    return None


def _to_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_int(value: object) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _parse_rate(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return _to_float(value)
    numerator, denominator = (_to_float(part) for part in value.split("/", 1))
    if not numerator or not denominator:
        return None
    return round(numerator / denominator, 3)


# --- Cadenas de filtros ----------------------------------------------------------------


def _denoise_chain(rnnoise_model: str | None, duration_s: float | None) -> str:
    if rnnoise_model is None:
        return f"{HIGHPASS},{FFT_DENOISE}"
    if not _SAFE_MODEL_NAME_RE.fullmatch(rnnoise_model):
        raise ValueError("Nombre de modelo RNNoise no permitido")
    # Ruta relativa: ffmpeg se ejecuta con cwd = carpeta del modelo (evita escapar rutas de Windows).
    rnnoise = f"arnndn=m={rnnoise_model},{FINITE_SAMPLES}"
    if not duration_s:
        return f"{HIGHPASS},{rnnoise}"
    return f"{HIGHPASS},{RNNOISE_TAIL_PAD},{rnnoise},atrim=duration={duration_s:.6f}"


def build_audio_filter(
    rnnoise_model: str | None, measurement: LoudnormMeasurement | None, duration_s: float | None = None
) -> str:
    """Ruido de fondo (RNNoise o FFT) + normalización EBU R128; segunda pasada lineal si hay medición.

    `duration_s` (duración del audio original) permite rellenar y recortar la cola de RNNoise.
    """
    loudnorm = f"loudnorm={LOUDNORM_TARGET}"
    if measurement is not None:
        loudnorm += (
            f":measured_I={measurement.input_i}:measured_TP={measurement.input_tp}"
            f":measured_LRA={measurement.input_lra}:measured_thresh={measurement.input_thresh}"
            f":offset={measurement.target_offset}:linear=true"
        )
    return f"{_denoise_chain(rnnoise_model, duration_s)},{loudnorm},aresample={OUTPUT_SAMPLE_RATE}"


def parse_loudnorm_output(stderr: str) -> LoudnormMeasurement | None:
    matches = _LOUDNORM_JSON_RE.findall(stderr)
    if not matches:
        return None
    try:
        raw = json.loads(matches[-1])
        measurement = LoudnormMeasurement(*(float(raw[key]) for key in _LOUDNORM_KEYS))
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    plausible = all(
        math.isfinite(value) and low <= value <= high
        for value, (low, high) in zip(astuple(measurement), _LOUDNORM_RANGES, strict=True)
    )
    return measurement if plausible else None


# --- Argumentos de ffmpeg -----------------------------------------------------------------


def _input_args(ffmpeg: str, src: Path, loglevel: str = "error") -> list[str]:
    return [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", loglevel, "-y", *INPUT_GUARD, "-i", str(src)]


def build_render_args(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    has_video: bool,
    has_audio: bool,
    video_filter: str | None,
    audio_filter: str | None,
) -> list[str]:
    args = _input_args(ffmpeg, src)
    if has_video:
        args += ["-map", "0:v:0", "-vf", video_filter or "null", *VIDEO_CODEC_ARGS]
    else:
        args += ["-vn"]
    if has_audio:
        args += ["-map", "0:a:0"]
        if audio_filter:
            args += ["-af", audio_filter]
        args += AAC_ARGS if has_video else FLAC_ARGS
    else:
        args += ["-an"]
    return [*args, "-map_metadata", "0", str(dst)]


def build_asr_args(ffmpeg: str, src: Path, dst: Path, audio_filter: str | None) -> list[str]:
    args = [*_input_args(ffmpeg, src), "-map", "0:a:0", "-vn"]
    if audio_filter:
        args += ["-af", audio_filter]
    return [*args, "-ac", "1", "-ar", str(ASR_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]


def portable_arguments(args: Sequence[str], src: Path, dst: Path) -> tuple[str, ...]:
    """Argumentos reproducibles para el linaje, sin rutas de esta máquina."""
    replacements = {str(src): "{input}", str(dst): "{output}"}
    return ("ffmpeg", *(replacements.get(arg, arg) for arg in args[1:]))


# --- Operaciones -----------------------------------------------------------------------------


def measure_loudness(
    tools: FfmpegTools, src: Path, *, rnnoise_model: str | None, duration_s: float | None, cwd: Path | None
) -> LoudnormMeasurement | None:
    """Primera pasada de loudnorm sobre el audio ya limpio (sin escribir salida)."""
    analysis = f"{_denoise_chain(rnnoise_model, duration_s)},loudnorm={LOUDNORM_TARGET}:print_format=json"
    args = [*_input_args(tools.ffmpeg, src, loglevel="info"), "-map", "0:a:0", "-af", analysis, "-f", "null", "-"]
    return parse_loudnorm_output(_run(args, timeout_s=tools.timeout_s, cwd=cwd).stderr)


def render(
    tools: FfmpegTools,
    src: Path,
    dst: Path,
    *,
    media: MediaInfo,
    video_filter: str | None,
    audio_filter: str | None,
    cwd: Path | None,
) -> RenderResult:
    args = build_render_args(
        tools.ffmpeg,
        src,
        dst,
        has_video=media.has_video,
        has_audio=media.has_audio,
        video_filter=video_filter,
        audio_filter=audio_filter,
    )
    _run(args, timeout_s=tools.timeout_s, cwd=cwd)
    return RenderResult(dst, portable_arguments(args, src, dst))


def extract_asr_audio(
    tools: FfmpegTools, src: Path, dst: Path, *, audio_filter: str | None, cwd: Path | None
) -> RenderResult:
    """Audio mono 16 kHz PCM para Whisper (con la misma cadena de mejora, sin pérdida intermedia)."""
    args = build_asr_args(tools.ffmpeg, src, dst, audio_filter)
    _run(args, timeout_s=tools.timeout_s, cwd=cwd)
    return RenderResult(dst, portable_arguments(args, src, dst))

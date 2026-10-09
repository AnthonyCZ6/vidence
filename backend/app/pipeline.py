"""Pipeline de procesamiento de una evidencia.

    original cifrado ─▶ descifrar a workspace efímero ─▶ verificar SHA-256 de ingesta
      ─▶ ffprobe (metadatos) ─▶ ffmpeg: vista previa sin filtros + versión mejorada
      ─▶ Whisper (audio mejorado, 16 kHz) ─▶ OpenCV (fotogramas del video mejorado)
      ─▶ cifrar cada derivado ─▶ manifiesto cifrado ─▶ borrar workspace

El original nunca se modifica: todo derivado registra el hash de su origen y
los argumentos exactos de ffmpeg que lo produjeron.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from app.frames import effective_interval, iter_frames
from app.media import (
    ENHANCE_VIDEO_FILTER,
    PREVIEW_VIDEO_FILTER,
    UPLOAD_MEDIA_TYPES,
    FfmpegTools,
    RenderResult,
    build_audio_filter,
    extract_asr_audio,
    measure_loudness,
    probe,
    render,
)
from app.schemas import (
    INGEST_NAME,
    MANIFEST_NAME,
    ORIGINAL_NAME,
    AssetRole,
    FrameAsset,
    IngestRecord,
    Manifest,
    MediaInfo,
    ProcessingRecord,
    StoredAsset,
    Transcript,
)
from app.transcribe import TRANSCRIPT_NOTICE, Transcriber
from app.vault import Vault
from app.vision import VisionDetector
from app.workspace import ephemeral_workspace

PIPELINE_NOTICES = (
    "El archivo original se conserva sin modificar; las versiones mejoradas son derivados con linaje registrado.",
    "Vidence no evalúa, juzga ni clasifica el contenido de la evidencia.",
)
VISION_NOTICE = (
    "Detecciones visuales generadas por IA local (YOLOv8). Representan probabilidades fácticas "
    "y no implican valoración de riesgo ni juicio sobre los objetos o personas detectadas."
)
TRANSCRIPTION_SOURCE = "enhanced_audio"
FRAMES_SOURCE = "enhanced_video"


class Stage(StrEnum):
    QUEUED = "queued"
    VERIFYING = "verifying"
    PROBING = "probing"
    ENHANCING = "enhancing"
    TRANSCRIBING = "transcribing"
    FRAMES = "frames"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"


ProgressReporter = Callable[[Stage, float | None], None]


class OriginalIntegrityError(Exception):
    """El original descifrado no coincide con el hash registrado al recibirlo."""


@dataclass(frozen=True)
class PipelineContext:
    vault: Vault
    tools: FfmpegTools
    transcriber: Transcriber
    workspace_root: Path
    rnnoise_dir: Path | None
    rnnoise_model: str | None
    frame_interval_s: float
    max_frames: int
    vision: VisionDetector | None = None

    @property
    def ffmpeg_cwd(self) -> Path | None:
        # arnndn recibe el modelo por nombre relativo: ffmpeg corre dentro de su carpeta.
        return self.rnnoise_dir if self.rnnoise_model else None


@dataclass(frozen=True)
class _Enhancement:
    derivatives: tuple[StoredAsset, ...]
    enhanced_path: Path
    audio_filter: str | None


def run_pipeline(context: PipelineContext, evidence_id: str, report: ProgressReporter) -> Manifest:
    ingest = IngestRecord.model_validate(context.vault.read_json(evidence_id, INGEST_NAME))
    if ingest.evidence_id != evidence_id:
        raise OriginalIntegrityError("El registro de ingesta no corresponde a esta evidencia")
    with ephemeral_workspace(context.workspace_root, evidence_id) as workspace:
        report(Stage.VERIFYING, None)
        original = _restore_original(context, ingest, workspace)

        report(Stage.PROBING, None)
        media = probe(context.tools, original)

        report(Stage.ENHANCING, None)
        enhancement = _enhance(context, ingest, workspace, original, media)

        transcript = None
        if media.has_audio:
            report(Stage.TRANSCRIBING, 0.0)
            transcript = _transcribe(context, workspace, original, enhancement.audio_filter, report)

        frames: tuple[FrameAsset, ...] = ()
        interval = None
        visual_summary: dict[str, int] = {}
        if media.has_video:
            report(Stage.FRAMES, 0.0)
            interval = effective_interval(context.frame_interval_s, media.duration_s, context.max_frames)
            frames, visual_summary = _store_frames(
                context, evidence_id, enhancement.enhanced_path, interval, media, report
            )

        report(Stage.FINALIZING, None)
        notices = list(PIPELINE_NOTICES)
        if transcript:
            notices.append(TRANSCRIPT_NOTICE)
        if frames and any(f.detections for f in frames):
            notices.append(VISION_NOTICE)

        manifest = Manifest(
            evidence_id=evidence_id,
            ingest=ingest,
            processed_at=datetime.now(UTC),
            integrity_verified=True,
            media=media,
            derivatives=enhancement.derivatives,
            transcript=transcript,
            frames=frames,
            frame_interval_s=interval,
            frames_source=FRAMES_SOURCE if media.has_video else None,
            visual_summary=visual_summary,
            notices=tuple(notices),
        )
        context.vault.store_json(evidence_id, MANIFEST_NAME, manifest.model_dump(mode="json"))
        return manifest


def _restore_original(context: PipelineContext, ingest: IngestRecord, workspace: Path) -> Path:
    suffix = Path(ingest.original_filename).suffix.lower()
    path = workspace / f"original{suffix if suffix in UPLOAD_MEDIA_TYPES else '.bin'}"
    restored = context.vault.decrypt_to_file(ingest.evidence_id, ORIGINAL_NAME, path)
    if restored.sha256 != ingest.original.sha256:
        raise OriginalIntegrityError("El SHA-256 del original no coincide con el registrado al recibirlo")
    return path


def _enhance(
    context: PipelineContext, ingest: IngestRecord, workspace: Path, original: Path, media: MediaInfo
) -> _Enhancement:
    cwd = context.ffmpeg_cwd
    audio_filter = None
    if media.has_audio:
        duration = media.audio_duration_s
        measurement = measure_loudness(
            context.tools, original, rnnoise_model=context.rnnoise_model, duration_s=duration, cwd=cwd
        )
        audio_filter = build_audio_filter(context.rnnoise_model, measurement, duration)
    extension, media_type = (".mp4", "video/mp4") if media.has_video else (".flac", "audio/flac")

    preview = render(
        context.tools,
        original,
        workspace / f"original_preview{extension}",
        media=media,
        video_filter=PREVIEW_VIDEO_FILTER,
        audio_filter=None,
        cwd=cwd,
    )
    enhanced = render(
        context.tools,
        original,
        workspace / f"enhanced{extension}",
        media=media,
        video_filter=ENHANCE_VIDEO_FILTER,
        audio_filter=audio_filter,
        cwd=cwd,
    )

    derivatives = tuple(
        _store_derivative(context, ingest, result, role, media_type)
        for result, role in ((preview, "original_preview"), (enhanced, "enhanced"))
    )
    return _Enhancement(derivatives, enhanced.output, audio_filter)


def _store_derivative(
    context: PipelineContext, ingest: IngestRecord, result: RenderResult, role: AssetRole, media_type: str
) -> StoredAsset:
    stored = context.vault.store_file(ingest.evidence_id, result.output.name, result.output)
    return StoredAsset(
        name=stored.name,
        role=role,
        media_type=media_type,
        sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        derived_from_sha256=ingest.original.sha256,
        processing=ProcessingRecord(tool="ffmpeg", tool_version=context.tools.version, arguments=result.arguments),
    )


def _transcribe(
    context: PipelineContext, workspace: Path, original: Path, audio_filter: str | None, report: ProgressReporter
) -> Transcript:
    asr = extract_asr_audio(
        context.tools, original, workspace / "asr_input.wav", audio_filter=audio_filter, cwd=context.ffmpeg_cwd
    )
    return context.transcriber.transcribe(
        asr.output,
        source_label=TRANSCRIPTION_SOURCE,
        on_progress=lambda progress: report(Stage.TRANSCRIBING, progress),
    )


def _store_frames(
    context: PipelineContext,
    evidence_id: str,
    video: Path,
    interval: float,
    media: MediaInfo,
    report: ProgressReporter,
) -> tuple[tuple[FrameAsset, ...], dict[str, int]]:
    frames: list[FrameAsset] = []
    visual_counts: dict[str, int] = {}
    for frame in iter_frames(video, interval_s=interval, max_frames=context.max_frames):
        name = f"frames/{frame.index:04d}.jpg"
        stored = context.vault.store_bytes(evidence_id, name, frame.jpeg)
        detections = ()
        if context.vision and frame.image_bgr is not None:
            detections = context.vision.detect(frame.image_bgr)
            for d in detections:
                visual_counts[d.label] = visual_counts.get(d.label, 0) + 1
        frames.append(
            FrameAsset(
                index=frame.index,
                timestamp_s=frame.timestamp_s,
                name=name,
                sha256=stored.sha256,
                size_bytes=stored.size_bytes,
                width=frame.width,
                height=frame.height,
                detections=detections,
            )
        )
        if media.duration_s:
            report(Stage.FRAMES, min(frame.timestamp_s / media.duration_s, 1.0))
    return tuple(frames), visual_counts

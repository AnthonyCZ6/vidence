"""Rutas HTTP bajo /api. Los routers son delgados: la lógica vive en ingest, pipeline y vault."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from app import offline_guard
from app.config import Settings
from app.ingest import EmptyUploadError, UnsupportedUploadError, UploadTooLargeError, ingest_stream
from app.jobs import JobManager, JobStatus
from app.media import FfmpegTools
from app.pipeline import Stage
from app.schemas import INGEST_NAME, MANIFEST_NAME, ORIGINAL_NAME, EvidenceSource, IngestRecord, Manifest
from app.transcribe import Transcriber
from app.vault import InvalidNameError, ObjectNotFoundError, Vault
from app.vision import VisionDetector

MAX_LISTED_EVIDENCE = 200
# Solo dígitos ASCII y longitud acotada: \d acepta dígitos Unicode e int() falla con miles de cifras.
_RANGE_RE = re.compile(r"bytes=([0-9]{0,15})-([0-9]{0,15})")

router = APIRouter(prefix="/api")


@dataclass(frozen=True)
class Services:
    settings: Settings
    vault: Vault
    jobs: JobManager
    tools: FfmpegTools
    transcriber: Transcriber
    vision: VisionDetector
    denoiser: str


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    offline_guard: bool
    network_policy: str
    whisper_model: str
    whisper_ready: bool
    vision_model: str
    vision_ready: bool
    compute: str
    denoiser: str
    ffmpeg: str


class UploadResponse(BaseModel):
    evidence_id: str
    job_id: str
    sha256: str
    size_bytes: int


class EvidenceSummary(BaseModel):
    evidence_id: str
    original_filename: str
    source: EvidenceSource
    received_at: datetime
    status: Literal["processed", "processing", "failed", "pending"]
    duration_s: float | None = None
    has_audio: bool | None = None
    has_video: bool | None = None


class RangeNotSatisfiableError(ValueError):
    """El rango HTTP pedido queda fuera del contenido."""


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Rango único `bytes=a-b` → (inicio, fin) inclusivo. Formatos no soportados → contenido completo."""
    match = _RANGE_RE.fullmatch(header.strip()) if header else None
    if match is None:
        return None
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return None
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length == 0:
            raise RangeNotSatisfiableError(header)
        return max(size - suffix_length, 0), size - 1
    start = int(start_text)
    end = min(int(end_text), size - 1) if end_text else size - 1
    if start >= size or start > end:
        raise RangeNotSatisfiableError(header)
    return start, end


# --- Estado ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        offline_guard=offline_guard.is_installed(),
        network_policy="Salida de red bloqueada: solo loopback",
        whisper_model=services.transcriber.model_name,
        whisper_ready=services.transcriber.is_available(),
        vision_model=services.vision.model_name,
        vision_ready=services.vision.is_available(),
        compute="CPU int8",
        denoiser=services.denoiser,
        ffmpeg=services.tools.version,
    )


# --- Ingesta y trabajos -------------------------------------------------------------------


@router.post("/evidence", status_code=201, response_model=UploadResponse)
async def upload_evidence(
    request: Request,
    services: ServicesDep,
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    source: Annotated[EvidenceSource, Query()] = "upload",
) -> UploadResponse:
    max_bytes = services.settings.max_upload_bytes
    declared_length = request.headers.get("content-length", "")
    if declared_length.isdigit() and int(declared_length) > max_bytes:
        raise HTTPException(413, f"El archivo supera el máximo de {max_bytes} bytes")
    try:
        record = await ingest_stream(
            services.vault, request.stream(), filename=filename, source=source, max_bytes=max_bytes
        )
    except UnsupportedUploadError as exc:
        raise HTTPException(415, str(exc)) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except EmptyUploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ClientDisconnect as exc:
        raise HTTPException(400, "La subida se interrumpió") from exc
    job = services.jobs.submit(record.evidence_id)
    return UploadResponse(
        evidence_id=record.evidence_id,
        job_id=job.job_id,
        sha256=record.original.sha256,
        size_bytes=record.original.size_bytes,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, services: ServicesDep) -> JobStatus:
    status = services.jobs.get(job_id)
    if status is None:
        raise HTTPException(404, "Trabajo no encontrado")
    return status


# --- Evidencias ----------------------------------------------------------------------------


@router.get("/evidence", response_model=list[EvidenceSummary])
def list_evidence(services: ServicesDep) -> list[EvidenceSummary]:
    summaries = (_summarize(services, eid) for eid in services.vault.list_evidence_ids()[:MAX_LISTED_EVIDENCE])
    return [summary for summary in summaries if summary is not None]


@router.get("/evidence/{evidence_id}", response_model=Manifest)
def get_manifest(evidence_id: str, services: ServicesDep) -> Manifest:
    return _load_manifest(services.vault, evidence_id)


@router.get("/evidence/{evidence_id}/assets/{name:path}")
def get_asset(evidence_id: str, name: str, request: Request, services: ServicesDep, download: bool = False) -> Response:
    manifest = _load_manifest(services.vault, evidence_id)
    media_type = manifest.media_type_for(name)
    if media_type is None:
        raise HTTPException(404, "Objeto no encontrado")
    size = services.vault.object_size(evidence_id, name)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    if download:
        filename = manifest.ingest.original_filename if name == ORIGINAL_NAME else name.rsplit("/", 1)[-1]
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
    try:
        byte_range = parse_range(request.headers.get("range"), size) if size else None
    except RangeNotSatisfiableError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if byte_range is None:
        return _stream(services.vault.iter_object(evidence_id, name), 200, media_type, headers, size)
    start, end = byte_range
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    body = services.vault.iter_object(evidence_id, name, byte_range)
    return _stream(body, 206, media_type, headers, end - start + 1)


@router.get("/evidence/{evidence_id}/report")
def get_report(evidence_id: str, services: ServicesDep) -> JSONResponse:
    manifest = _load_manifest(services.vault, evidence_id)
    report = {
        "report": "vidence-evidence-report",
        "report_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": manifest.model_dump(mode="json"),
    }
    headers = {
        "Content-Disposition": f'attachment; filename="vidence-{evidence_id[:8]}.json"',
        "Cache-Control": "no-store",
    }
    return JSONResponse(report, headers=headers)


# --- Auxiliares -----------------------------------------------------------------------------


def _stream(body: Iterator[bytes], status: int, media_type: str, headers: dict[str, str], length: int) -> Response:
    return StreamingResponse(
        body, status_code=status, media_type=media_type, headers={**headers, "Content-Length": str(length)}
    )


def _load_manifest(vault: Vault, evidence_id: str) -> Manifest:
    try:
        manifest = Manifest.model_validate(vault.read_json(evidence_id, MANIFEST_NAME))
    except InvalidNameError as exc:
        raise HTTPException(404, "Evidencia no encontrada") from exc
    except ObjectNotFoundError as exc:
        if vault.exists(evidence_id, INGEST_NAME):
            raise HTTPException(409, "La evidencia aún no tiene resultados (en proceso o con error)") from exc
        raise HTTPException(404, "Evidencia no encontrada") from exc
    if manifest.evidence_id != evidence_id or manifest.ingest.evidence_id != evidence_id:
        raise HTTPException(500, "El manifiesto no corresponde a esta evidencia")
    return manifest


def _summarize(services: Services, evidence_id: str) -> EvidenceSummary | None:
    vault = services.vault
    try:
        ingest = IngestRecord.model_validate(vault.read_json(evidence_id, INGEST_NAME))
    except ObjectNotFoundError:
        return None  # subida incompleta: no hay registro de ingesta
    base: dict[str, Any] = {
        "evidence_id": evidence_id,
        "original_filename": ingest.original_filename,
        "source": ingest.source,
        "received_at": ingest.received_at,
    }
    if vault.exists(evidence_id, MANIFEST_NAME):
        media = Manifest.model_validate(vault.read_json(evidence_id, MANIFEST_NAME)).media
        return EvidenceSummary(
            **base,
            status="processed",
            duration_s=media.duration_s,
            has_audio=media.has_audio,
            has_video=media.has_video,
        )
    job = services.jobs.latest_for_evidence(evidence_id)
    if job is None:
        return EvidenceSummary(**base, status="pending")
    return EvidenceSummary(**base, status="failed" if job.stage == Stage.FAILED else "processing")

"""Construcción de la aplicación FastAPI (sin efectos de red: todo es local)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api import Services, router
from app.config import Settings
from app.crypto import CryptoError
from app.jobs import JobManager
from app.media import FfmpegTools
from app.model_files import verify_model_files
from app.pipeline import PipelineContext, run_pipeline
from app.transcribe import Transcriber, WhisperTranscriber
from app.vault import InvalidNameError, ObjectNotFoundError, Vault
from app.vision import VisionDetector, YoloVisionDetector
from app.workspace import purge_stale_workspaces

logger = logging.getLogger(__name__)

# Sin CDNs ni orígenes externos: el navegador tampoco puede sacar datos del equipo.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; "
        "script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(self), microphone=(self), geolocation=(), interest-cohort=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "vidence"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _is_same_origin_write(headers: Headers) -> bool:
    """Una cabecera personalizada obliga a preflight CORS (que nunca aprobamos) en peticiones ajenas."""
    if headers.get(CSRF_HEADER) != CSRF_VALUE:
        return False
    origin = headers.get("origin")
    return origin is None or urlsplit(origin).netloc == headers.get("host")


class SecurityMiddleware:
    """Cabeceras de seguridad en toda respuesta y bloqueo CSRF de escrituras."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).update(SECURITY_HEADERS)
            await send(message)

        if scope["method"] not in SAFE_METHODS and not _is_same_origin_write(Headers(scope=scope)):
            rejection = JSONResponse({"detail": "Solicitud rechazada: origen no permitido"}, status_code=403)
            await rejection(scope, receive, send_with_headers)
            return
        await self._app(scope, receive, send_with_headers)


def _resolve_rnnoise(settings: Settings) -> str | None:
    if not settings.rnnoise_path.is_file():
        logger.warning("Modelo RNNoise no aprovisionado: se usará el filtro FFT (afftdn)")
        return None
    verify_model_files(settings.models_dir, f"rnnoise/{settings.rnnoise_model}")
    return settings.rnnoise_model


def _register_error_handlers(app: FastAPI) -> None:
    async def not_found(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "No encontrado"}, status_code=404)

    async def integrity_failure(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Falló la verificación de un objeto cifrado: %s", exc)
        return JSONResponse({"detail": "Falló la verificación de integridad del contenido cifrado"}, status_code=500)

    app.add_exception_handler(InvalidNameError, not_found)
    app.add_exception_handler(ObjectNotFoundError, not_found)
    app.add_exception_handler(CryptoError, integrity_failure)


def create_app(
    settings: Settings,
    vault: Vault,
    *,
    transcriber: Transcriber | None = None,
    tools: FfmpegTools | None = None,
    vision: VisionDetector | None = None,
) -> FastAPI:
    tools = tools or FfmpegTools.resolve(settings.ffmpeg_bin, settings.ffprobe_bin, settings.ffmpeg_timeout_seconds)
    transcriber = transcriber or WhisperTranscriber(
        models_root=settings.models_dir,
        model_name=settings.whisper_model,
        language=settings.whisper_language,
        cpu_threads=settings.whisper_cpu_threads,
        beam_size=settings.whisper_beam_size,
    )
    vision = vision or YoloVisionDetector(
        models_root=settings.models_dir,
        model_name=settings.vision_model,
        confidence_threshold=settings.vision_confidence_threshold,
        nms_threshold=settings.vision_nms_threshold,
    )
    rnnoise_model = _resolve_rnnoise(settings)
    if purged := purge_stale_workspaces(settings.workspace_dir):
        logger.warning("Se eliminaron %d workspaces en claro de una ejecución anterior", purged)

    context = PipelineContext(
        vault=vault,
        tools=tools,
        transcriber=transcriber,
        workspace_root=settings.workspace_dir,
        rnnoise_dir=settings.rnnoise_dir if rnnoise_model else None,
        rnnoise_model=rnnoise_model,
        frame_interval_s=settings.frame_interval_seconds,
        max_frames=settings.max_frames,
        vision=vision,
    )
    jobs = JobManager(partial(run_pipeline, context))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            jobs.shutdown()

    app = FastAPI(title="Vidence", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.services = Services(
        settings=settings,
        vault=vault,
        jobs=jobs,
        tools=tools,
        transcriber=transcriber,
        vision=vision,
        denoiser="rnnoise" if rnnoise_model else "afftdn",
    )
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.trusted_hosts))
    _register_error_handlers(app)
    app.include_router(router)
    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app

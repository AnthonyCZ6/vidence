"""Ingesta de evidencia en streaming: SHA-256 y cifrado en una sola pasada.

Los bytes recibidos nunca se escriben en claro. A diferencia de `UploadFile`
de Starlette (que vuelca a %TEMP% en claro todo archivo de más de 1 MB), aquí
cada fragmento se cifra al llegar. El nombre del cliente solo se guarda como
metadato cifrado; jamás se usa como ruta.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from app.media import UPLOAD_MEDIA_TYPES
from app.schemas import INGEST_NAME, ORIGINAL_NAME, EvidenceSource, IngestRecord, StoredAsset
from app.vault import Vault

MAX_FILENAME_CHARS = 200
# Control (Cc), formato invisible como U+202E que invierte el texto (Cf) y separadores de línea.
_STRIPPED_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


class IngestError(Exception):
    """Error base de ingesta."""


class UnsupportedUploadError(IngestError):
    """Extensión o nombre de archivo no admitido."""


class UploadTooLargeError(IngestError):
    """El archivo supera el tamaño máximo configurado."""


class EmptyUploadError(IngestError):
    """No se recibió ningún byte."""


def sanitize_filename(raw: str) -> str:
    """Nombre base sin rutas ni caracteres de control/invisibles, con extensión multimedia admitida."""
    visible = "".join(ch for ch in raw if unicodedata.category(ch) not in _STRIPPED_CATEGORIES)
    cleaned = visible.replace("\\", "/")
    name = cleaned.rsplit("/", 1)[-1].strip()
    suffix = Path(name).suffix
    if not name or suffix.lower() not in UPLOAD_MEDIA_TYPES:
        allowed = ", ".join(sorted(UPLOAD_MEDIA_TYPES))
        raise UnsupportedUploadError(f"Tipo de archivo no admitido. Formatos aceptados: {allowed}")
    if len(name) > MAX_FILENAME_CHARS:
        name = name[: MAX_FILENAME_CHARS - len(suffix)] + suffix
    return name


async def ingest_stream(
    vault: Vault,
    chunks: AsyncIterator[bytes],
    *,
    filename: str,
    source: EvidenceSource,
    max_bytes: int,
) -> IngestRecord:
    original_filename = sanitize_filename(filename)
    media_type = UPLOAD_MEDIA_TYPES[Path(original_filename).suffix.lower()]
    evidence_id = vault.new_evidence_id()
    writer = vault.open_writer(evidence_id, ORIGINAL_NAME)
    try:
        async for chunk in chunks:
            if writer.size_bytes + len(chunk) > max_bytes:
                raise UploadTooLargeError(f"El archivo supera el máximo de {max_bytes} bytes")
            if chunk:
                await asyncio.to_thread(writer.write, chunk)
        if writer.size_bytes == 0:
            raise EmptyUploadError("El archivo está vacío")
        stored = await asyncio.to_thread(writer.commit)
    except BaseException:
        # Incluye la desconexión del cliente (CancelledError): no deben quedar restos.
        writer.abort()
        vault.discard_if_empty(evidence_id)
        raise

    record = IngestRecord(
        evidence_id=evidence_id,
        original_filename=original_filename,
        source=source,
        received_at=datetime.now(UTC),
        original=StoredAsset(
            name=ORIGINAL_NAME,
            role="original",
            media_type=media_type,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
        ),
    )
    await asyncio.to_thread(vault.store_json, evidence_id, INGEST_NAME, record.model_dump(mode="json"))
    return record

"""Ingesta en streaming: hash + cifrado en una sola pasada, sin tocar el disco en claro."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator

import pytest

from app.ingest import EmptyUploadError, UnsupportedUploadError, UploadTooLargeError, ingest_stream, sanitize_filename
from app.schemas import INGEST_NAME, ORIGINAL_NAME, IngestRecord
from app.vault import Vault


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def _ingest(vault: Vault, *parts: bytes, filename: str = "grabación.m4a", max_bytes: int = 10_000) -> IngestRecord:
    return asyncio.run(ingest_stream(vault, _chunks(*parts), filename=filename, source="upload", max_bytes=max_bytes))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("video.MP4", "video.MP4"),
        ("C:\\Users\\alguien\\clip.mov", "clip.mov"),
        ("../../etc/audio.wav", "audio.wav"),
        ("nota\x00.mp3", "nota.mp3"),
        ("  voz.webm ", "voz.webm"),
        ("evidencia‮gpj.mp4", "evidenciagpj.mp4"),
        ("nota​ .wav", "nota.wav"),
    ],
)
def test_sanitize_filename_strips_paths_and_control_chars(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["documento.pdf", "script.exe", "sin_extension", "", "   ", "../"])
def test_sanitize_filename_rejects_unsupported_files(raw: str) -> None:
    with pytest.raises(UnsupportedUploadError):
        sanitize_filename(raw)


def test_long_filenames_are_truncated_keeping_extension() -> None:
    name = sanitize_filename("a" * 300 + ".mp4")
    assert len(name) <= 200
    assert name.endswith(".mp4")


def test_ingest_stores_encrypted_original_and_record(vault: Vault) -> None:
    record = _ingest(vault, b"parte-1|", b"parte-2")

    assert record.original.sha256 == hashlib.sha256(b"parte-1|parte-2").hexdigest()
    assert record.original.media_type == "audio/mp4"
    assert record.original_filename == "grabación.m4a"
    assert vault.read_bytes(record.evidence_id, ORIGINAL_NAME) == b"parte-1|parte-2"
    assert IngestRecord.model_validate(vault.read_json(record.evidence_id, INGEST_NAME)) == record


def test_oversized_upload_is_rejected_without_leftovers(vault: Vault) -> None:
    with pytest.raises(UploadTooLargeError):
        _ingest(vault, b"x" * 600, b"x" * 600, max_bytes=1000)
    assert vault.list_evidence_ids() == []


def test_empty_upload_is_rejected_without_leftovers(vault: Vault) -> None:
    with pytest.raises(EmptyUploadError):
        _ingest(vault, b"")
    assert vault.list_evidence_ids() == []

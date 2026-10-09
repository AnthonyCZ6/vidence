"""Pipeline completo: original cifrado → verificación → mejora → IA → derivados cifrados."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.ingest import ingest_stream
from app.media import FfmpegTools, UnsupportedMediaError
from app.pipeline import OriginalIntegrityError, PipelineContext, Stage, run_pipeline
from app.schemas import INGEST_NAME, MANIFEST_NAME, IngestRecord, Manifest
from app.vault import Vault
from tests.conftest import FakeTranscriber, FakeVisionDetector, MediaSamples

pytestmark = pytest.mark.ffmpeg


@pytest.fixture
def context(vault: Vault, tools: FfmpegTools, rnnoise_dir: Path, tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        vault=vault,
        tools=tools,
        transcriber=FakeTranscriber(),
        workspace_root=tmp_path / "workspace",
        rnnoise_dir=rnnoise_dir,
        rnnoise_model="cb.rnnn",
        frame_interval_s=1.0,
        max_frames=50,
        vision=FakeVisionDetector(),
    )


def _ingest_file(vault: Vault, path: Path, filename: str) -> IngestRecord:
    async def chunks() -> AsyncIterator[bytes]:
        yield path.read_bytes()

    return asyncio.run(ingest_stream(vault, chunks(), filename=filename, source="upload", max_bytes=10**9))


def _run(context: PipelineContext, evidence_id: str) -> Manifest:
    return run_pipeline(context, evidence_id, lambda stage, progress: None)


def test_audio_evidence_gets_flac_derivatives_and_transcript(
    context: PipelineContext, media_samples: MediaSamples
) -> None:
    record = _ingest_file(context.vault, media_samples.noisy_audio, "ruido.wav")
    stages: list[Stage] = []

    manifest = run_pipeline(context, record.evidence_id, lambda stage, progress: stages.append(stage))

    roles = {asset.role: asset for asset in manifest.derivatives}
    assert set(roles) == {"original_preview", "enhanced"}
    assert all(a.media_type == "audio/flac" for a in manifest.derivatives)
    assert all(a.derived_from_sha256 == record.original.sha256 for a in manifest.derivatives)
    assert roles["enhanced"].processing is not None
    assert any("arnndn=m=cb.rnnn" in arg for arg in roles["enhanced"].processing.arguments)
    assert manifest.integrity_verified
    assert manifest.transcript is not None
    assert manifest.frames == ()
    assert stages[0] == Stage.VERIFYING
    assert Stage.TRANSCRIBING in stages
    assert stages[-1] == Stage.FINALIZING
    assert not (context.workspace_root / record.evidence_id).exists()


def test_video_evidence_gets_mp4_derivatives_and_frames(context: PipelineContext, media_samples: MediaSamples) -> None:
    record = _ingest_file(context.vault, media_samples.video, "clip.mp4")

    manifest = _run(context, record.evidence_id)

    assert {a.media_type for a in manifest.derivatives} == {"video/mp4"}
    assert len(manifest.frames) == 4
    assert manifest.frames_source == "enhanced_video"
    assert manifest.frames[0].detections
    assert manifest.frames[0].detections[0].label == "persona"
    assert manifest.visual_summary.get("persona") == 4
    assert context.vault.read_bytes(record.evidence_id, manifest.frames[0].name).startswith(b"\xff\xd8")
    assert Manifest.model_validate(context.vault.read_json(record.evidence_id, MANIFEST_NAME)) == manifest


def test_stored_derivatives_match_their_recorded_hashes(context: PipelineContext, media_samples: MediaSamples) -> None:
    record = _ingest_file(context.vault, media_samples.video, "clip.mp4")
    manifest = _run(context, record.evidence_id)
    for asset in manifest.derivatives:
        content = context.vault.read_bytes(record.evidence_id, asset.name)
        assert hashlib.sha256(content).hexdigest() == asset.sha256


def test_silent_video_skips_transcription(context: PipelineContext, media_samples: MediaSamples) -> None:
    record = _ingest_file(context.vault, media_samples.silent_video, "mudo.mp4")
    manifest = _run(context, record.evidence_id)
    assert manifest.transcript is None
    assert manifest.frames


def test_fft_denoiser_is_used_when_rnnoise_is_missing(context: PipelineContext, media_samples: MediaSamples) -> None:
    fallback = dataclasses.replace(context, rnnoise_model=None, rnnoise_dir=None)
    record = _ingest_file(fallback.vault, media_samples.noisy_audio, "ruido.wav")
    enhanced = next(a for a in _run(fallback, record.evidence_id).derivatives if a.role == "enhanced")
    assert enhanced.processing is not None
    assert any("afftdn" in arg for arg in enhanced.processing.arguments)


def test_tampered_ingest_hash_aborts_processing(context: PipelineContext, media_samples: MediaSamples) -> None:
    record = _ingest_file(context.vault, media_samples.noisy_audio, "ruido.wav")
    forged_original = record.original.model_copy(update={"sha256": "0" * 64})
    forged = record.model_copy(update={"original": forged_original})
    context.vault.store_json(record.evidence_id, INGEST_NAME, forged.model_dump(mode="json"))

    with pytest.raises(OriginalIntegrityError):
        _run(context, record.evidence_id)
    assert not context.vault.exists(record.evidence_id, MANIFEST_NAME)
    assert not (context.workspace_root / record.evidence_id).exists()


def test_content_that_is_not_media_is_rejected(context: PipelineContext, tmp_path: Path) -> None:
    fake = tmp_path / "falso.mp4"
    fake.write_bytes(b"esto no es un video " * 100)
    record = _ingest_file(context.vault, fake, "falso.mp4")
    with pytest.raises(UnsupportedMediaError):
        _run(context, record.evidence_id)

"""API HTTP de punta a punta (TestClient) con ffmpeg real y transcriptor simulado."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import RangeNotSatisfiableError, parse_range
from app.config import Settings
from app.ingest import ingest_stream
from app.main import create_app
from app.media import FfmpegTools
from app.vault import Vault
from tests.conftest import FAST_KDF, TEST_PASSPHRASE, FakeTranscriber, FakeVisionDetector, MediaSamples

CSRF = {"X-Requested-With": "vidence"}
JOB_TIMEOUT_SECONDS = 120


# --- parse_range (puro) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("bytes=0-9", (0, 9)),
        ("bytes=90-", (90, 99)),
        ("bytes=-10", (90, 99)),
        ("bytes=-500", (0, 99)),
        ("bytes=50-500", (50, 99)),
        ("items=0-1", None),
        ("bytes=0-1,5-6", None),
        ("bytes=-", None),
        ("bytes=" + "9" * 5000 + "-", None),
        ("bytes=١-٢", None),
    ],
)
def test_parse_range(header: str | None, expected: tuple[int, int] | None) -> None:
    assert parse_range(header, 100) == expected


@pytest.mark.parametrize("header", ["bytes=100-", "bytes=5-2", "bytes=-0"])
def test_parse_range_unsatisfiable(header: str) -> None:
    with pytest.raises(RangeNotSatisfiableError):
        parse_range(header, 100)


# --- Cliente HTTP -----------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, settings: Settings, tools: FfmpegTools) -> Iterator[TestClient]:
    test_settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=settings.models_dir,
        frontend_dir=settings.frontend_dir,
        trusted_hosts=("testserver",),
        max_upload_bytes=5_000_000,
        frame_interval_seconds=1.0,
        ffmpeg_bin=settings.ffmpeg_bin,
        ffprobe_bin=settings.ffprobe_bin,
    )
    vault = Vault.open_or_create(test_settings.vault_dir, TEST_PASSPHRASE, kdf_cost=FAST_KDF)
    app = create_app(test_settings, vault, transcriber=FakeTranscriber(), tools=tools, vision=FakeVisionDetector())
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, content: bytes, filename: str) -> Any:
    return client.post(
        "/api/evidence",
        params={"filename": filename, "source": "upload"},
        content=content,
        headers={**CSRF, "Content-Type": "application/octet-stream"},
    )


def _wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["stage"] in ("done", "failed"):
            return status
        time.sleep(0.1)
    raise AssertionError("el trabajo no terminó a tiempo")


@pytest.fixture
def processed_video(client: TestClient, media_samples: MediaSamples) -> dict[str, Any]:
    upload = _upload(client, media_samples.video.read_bytes(), "clip.mp4")
    assert upload.status_code == 201, upload.text
    status = _wait_for_job(client, upload.json()["job_id"])
    assert status["stage"] == "done", status
    return client.get(f"/api/evidence/{upload.json()['evidence_id']}").json()


def _asset_url(manifest: dict[str, Any], role: str) -> str:
    asset = next(a for a in manifest["derivatives"] if a["role"] == role)
    return f"/api/evidence/{manifest['evidence_id']}/assets/{asset['name']}"


# --- Seguridad --------------------------------------------------------------------------


def test_health_reports_the_offline_ai_stack(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["offline_guard"] is True
    assert body["compute"] == "CPU int8"
    assert body["denoiser"] == "rnnoise"
    assert body["vision_model"] == "fake-yolo"
    assert body["vision_ready"] is True
    assert body["ffmpeg"].startswith("ffmpeg version")


def test_security_headers_are_sent(client: TestClient) -> None:
    headers = client.get("/api/health").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert "connect-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_disabled_because_they_load_cdn_assets(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_untrusted_host_header_is_rejected(client: TestClient) -> None:
    assert client.get("/api/health", headers={"Host": "rebinding.example"}).status_code == 400


def test_write_without_csrf_header_is_rejected(client: TestClient) -> None:
    response = client.post("/api/evidence", params={"filename": "a.wav"}, content=b"x")
    assert response.status_code == 403


def test_cross_origin_write_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/evidence",
        params={"filename": "a.wav"},
        content=b"x",
        headers={**CSRF, "Origin": "http://evil.example"},
    )
    assert response.status_code == 403


# --- Subida ---------------------------------------------------------------------------------


def test_unsupported_extension_is_rejected(client: TestClient) -> None:
    assert _upload(client, b"MZ", "programa.exe").status_code == 415


def test_empty_upload_is_rejected(client: TestClient) -> None:
    assert _upload(client, b"", "vacio.wav").status_code == 400


def test_oversized_upload_is_rejected(client: TestClient) -> None:
    assert _upload(client, b"x" * 5_000_001, "grande.wav").status_code == 413


def test_upload_returns_hash_of_received_bytes(client: TestClient, media_samples: MediaSamples) -> None:
    content = media_samples.noisy_audio.read_bytes()
    body = _upload(client, content, "ruido.wav").json()
    assert body["sha256"] == hashlib.sha256(content).hexdigest()
    assert body["size_bytes"] == len(content)
    _wait_for_job(client, body["job_id"])


def test_failed_processing_is_reported(client: TestClient) -> None:
    body = _upload(client, b"esto no es un video " * 50, "falso.mp4").json()
    status = _wait_for_job(client, body["job_id"])
    assert status["stage"] == "failed"
    assert "Formato no compatible" in status["error"]
    listed = {item["evidence_id"]: item for item in client.get("/api/evidence").json()}
    assert listed[body["evidence_id"]]["status"] == "failed"


# --- Resultados ------------------------------------------------------------------------------------


def test_processed_video_has_frames_transcript_and_lineage(processed_video: dict[str, Any]) -> None:
    assert len(processed_video["frames"]) == 4
    assert processed_video["transcript"]["segments"][0]["text"] == "hola"
    assert processed_video["integrity_verified"] is True
    for asset in processed_video["derivatives"]:
        assert asset["derived_from_sha256"] == processed_video["ingest"]["original"]["sha256"]


def test_evidence_list_includes_processed_item(client: TestClient, processed_video: dict[str, Any]) -> None:
    listed = {item["evidence_id"]: item for item in client.get("/api/evidence").json()}
    item = listed[processed_video["evidence_id"]]
    assert item["status"] == "processed"
    assert item["original_filename"] == "clip.mp4"
    assert item["has_video"] is True


def test_full_asset_download_matches_recorded_hash(client: TestClient, processed_video: dict[str, Any]) -> None:
    response = client.get(_asset_url(processed_video, "enhanced"))
    enhanced = next(a for a in processed_video["derivatives"] if a["role"] == "enhanced")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["accept-ranges"] == "bytes"
    assert hashlib.sha256(response.content).hexdigest() == enhanced["sha256"]


def test_range_requests_return_partial_content(client: TestClient, processed_video: dict[str, Any]) -> None:
    url = _asset_url(processed_video, "enhanced")
    full = client.get(url).content

    middle = client.get(url, headers={"Range": "bytes=100-1099"})
    suffix = client.get(url, headers={"Range": "bytes=-50"})

    assert middle.status_code == 206
    assert middle.content == full[100:1100]
    assert middle.headers["content-range"] == f"bytes 100-1099/{len(full)}"
    assert suffix.content == full[-50:]


def test_unsatisfiable_range_returns_416(client: TestClient, processed_video: dict[str, Any]) -> None:
    response = client.get(_asset_url(processed_video, "enhanced"), headers={"Range": "bytes=999999999-"})
    assert response.status_code == 416


def test_frames_are_served_as_jpeg(client: TestClient, processed_video: dict[str, Any]) -> None:
    frame = processed_video["frames"][0]
    response = client.get(f"/api/evidence/{processed_video['evidence_id']}/assets/{frame['name']}")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_original_download_keeps_original_filename(client: TestClient, processed_video: dict[str, Any]) -> None:
    response = client.get(
        f"/api/evidence/{processed_video['evidence_id']}/assets/original.bin", params={"download": "true"}
    )
    assert "clip.mp4" in response.headers["content-disposition"]
    assert hashlib.sha256(response.content).hexdigest() == processed_video["ingest"]["original"]["sha256"]


@pytest.mark.parametrize("name", ["../vault.json", "manifest.json", "ingest.json", "frames/9999.jpg"])
def test_objects_outside_the_manifest_are_not_served(
    client: TestClient, processed_video: dict[str, Any], name: str
) -> None:
    assert client.get(f"/api/evidence/{processed_video['evidence_id']}/assets/{name}").status_code == 404


def test_report_is_downloadable_json(client: TestClient, processed_video: dict[str, Any]) -> None:
    response = client.get(f"/api/evidence/{processed_video['evidence_id']}/report")
    assert response.headers["content-disposition"].startswith("attachment")
    body = response.json()
    assert body["report"] == "vidence-evidence-report"
    assert body["manifest"]["evidence_id"] == processed_video["evidence_id"]


@pytest.mark.parametrize("evidence_id", ["no-es-un-id", "0" * 32])
def test_unknown_evidence_returns_404(client: TestClient, evidence_id: str) -> None:
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 404


def test_unprocessed_evidence_returns_409(client: TestClient) -> None:
    vault: Vault = client.app.state.services.vault  # type: ignore[attr-defined]

    async def chunks() -> AsyncIterator[bytes]:
        yield b"datos"

    record = asyncio.run(ingest_stream(vault, chunks(), filename="a.wav", source="upload", max_bytes=100))

    assert client.get(f"/api/evidence/{record.evidence_id}").status_code == 409
    listed = {item["evidence_id"]: item for item in client.get("/api/evidence").json()}
    assert listed[record.evidence_id]["status"] == "pending"

"""Cola de trabajos en segundo plano (un solo worker)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import pytest

from app.crypto import IntegrityError
from app.jobs import JobManager, JobStatus, describe_failure
from app.media import FfmpegError, UnsupportedMediaError
from app.pipeline import OriginalIntegrityError, Stage

EVIDENCE_ID = "a" * 32
TERMINAL = (Stage.DONE, Stage.FAILED)


def _wait_for(manager: JobManager, job_id: str, condition: Callable[[JobStatus], bool]) -> JobStatus:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = manager.get(job_id)
        if status is not None and condition(status):
            return status
        time.sleep(0.01)
    raise AssertionError("el trabajo no alcanzó el estado esperado")


@pytest.fixture
def make_manager() -> Iterator[Callable[[Callable[..., object]], JobManager]]:
    managers: list[JobManager] = []

    def factory(runner: Callable[..., object]) -> JobManager:
        managers.append(JobManager(runner))
        return managers[-1]

    yield factory
    for manager in managers:
        manager.shutdown()


def test_successful_job_reports_done(make_manager: Callable[..., JobManager]) -> None:
    processed: list[str] = []

    def runner(evidence_id: str, report: Callable[[Stage, float | None], None]) -> None:
        report(Stage.ENHANCING, None)
        processed.append(evidence_id)

    manager = make_manager(runner)
    job = manager.submit(EVIDENCE_ID)
    final = _wait_for(manager, job.job_id, lambda s: s.stage in TERMINAL)

    assert job.stage == Stage.QUEUED
    assert (final.stage, final.progress, final.error) == (Stage.DONE, 1.0, None)
    assert processed == [EVIDENCE_ID]


def test_job_records_every_executed_stage_even_if_brief(make_manager: Callable[..., JobManager]) -> None:
    """La interfaz consulta cada 700 ms: una etapa breve no debe mostrarse como omitida."""

    def runner(evidence_id: str, report: Callable[[Stage, float | None], None]) -> None:
        for stage in (Stage.VERIFYING, Stage.PROBING, Stage.ENHANCING, Stage.TRANSCRIBING):
            report(stage, None)
        report(Stage.TRANSCRIBING, 0.5)
        report(Stage.FINALIZING, None)

    manager = make_manager(runner)
    final = _wait_for(manager, manager.submit(EVIDENCE_ID).job_id, lambda s: s.stage in TERMINAL)

    assert final.stages_seen == (Stage.VERIFYING, Stage.PROBING, Stage.ENHANCING, Stage.TRANSCRIBING, Stage.FINALIZING)


def test_failed_job_exposes_only_a_safe_message(make_manager: Callable[..., JobManager]) -> None:
    def runner(evidence_id: str, report: Callable[[Stage, float | None], None]) -> None:
        raise RuntimeError("detalle interno con ruta C:/secreto")

    manager = make_manager(runner)
    final = _wait_for(manager, manager.submit(EVIDENCE_ID).job_id, lambda s: s.stage in TERMINAL)

    assert final.stage == Stage.FAILED
    assert final.error is not None
    assert "C:/secreto" not in final.error


def test_progress_is_visible_while_running(make_manager: Callable[..., JobManager]) -> None:
    release = threading.Event()

    def runner(evidence_id: str, report: Callable[[Stage, float | None], None]) -> None:
        report(Stage.TRANSCRIBING, 0.4)
        release.wait(5)

    manager = make_manager(runner)
    job = manager.submit(EVIDENCE_ID)
    running = _wait_for(manager, job.job_id, lambda s: s.stage == Stage.TRANSCRIBING)
    release.set()

    assert running.progress == 0.4
    assert "Whisper" in running.message


def test_latest_job_for_evidence(make_manager: Callable[..., JobManager]) -> None:
    manager = make_manager(lambda evidence_id, report: None)
    manager.submit(EVIDENCE_ID)
    second = manager.submit(EVIDENCE_ID)
    latest = manager.latest_for_evidence(EVIDENCE_ID)
    assert latest is not None
    assert latest.job_id == second.job_id
    assert manager.latest_for_evidence("b" * 32) is None


def test_unknown_job_returns_none(make_manager: Callable[..., JobManager]) -> None:
    assert make_manager(lambda evidence_id, report: None).get("inexistente") is None


@pytest.mark.parametrize(
    ("exc", "fragment"),
    [
        (UnsupportedMediaError("Contenedor no permitido: hls"), "hls"),
        (FfmpegError("fallo", "stderr"), "ffmpeg"),
        (OriginalIntegrityError("x"), "integridad"),
        (IntegrityError("x"), "integridad"),
        (ValueError("interno"), "registro local"),
    ],
)
def test_describe_failure(exc: Exception, fragment: str) -> None:
    assert fragment in describe_failure(exc)

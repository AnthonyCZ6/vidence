"""Cadena de suministro: ningún SDK de nube, analítica o exportador de telemetría instalado."""

from __future__ import annotations

import importlib.metadata

import pytest

from app.config import BACKEND_ROOT

FORBIDDEN_DISTRIBUTIONS = (
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-exporter-otlp-proto-grpc",
    "fastapi-cloud-cli",
    "sentry-sdk",
    "posthog",
    "mixpanel",
    "boto3",
    "google-cloud-storage",
    "google-cloud-speech",
    "azure-cognitiveservices-speech",
    "openai",
    "anthropic",
    "assemblyai",
    "deepgram-sdk",
)


@pytest.mark.parametrize("distribution", FORBIDDEN_DISTRIBUTIONS)
def test_no_cloud_or_telemetry_sdk_is_installed(distribution: str) -> None:
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution(distribution)


def test_runtime_requirements_use_no_extras() -> None:
    """Los extras (p. ej. fastapi 'standard') traen exportadores OTLP y CLIs de nube."""
    lines = (BACKEND_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line.split("#", 1)[0].strip() for line in lines]
    assert not any("[" in requirement for requirement in requirements if requirement)

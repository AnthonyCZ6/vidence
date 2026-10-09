"""Fixtures compartidas.

Importar `app` instala el guardia offline para toda la sesión de tests: si
alguna dependencia intenta salir a la red, el test que la dispare falla.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import app  # noqa: F401  (efecto buscado: instala el guardia offline)
from app.config import Settings
from app.crypto import KdfCost
from app.media import FfmpegTools, MediaError
from app.schemas import Segment, Transcript
from app.vault import Vault
from app.vision import FakeVisionDetector

TEST_PASSPHRASE = "frase-de-prueba-larga"
FAST_KDF = KdfCost(iterations=1, lanes=1, memory_cost_kib=8 * 1024)
TEST_CHUNK_SIZE = 1024
SPEECH_TEXT = "Hello. This is a test of the local transcription system. The recording was made on Tuesday."
GENERATION_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class MediaSamples:
    noisy_audio: Path
    video: Path
    silent_video: Path
    webm_audio: Path


class FakeTranscriber:
    """Transcriptor determinista para tests de pipeline/API (Whisper real se prueba en test_transcribe)."""

    model_name = "fake"

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_path: Path, *, source_label: str, on_progress: Callable[[float], None]) -> Transcript:
        if not audio_path.is_file():
            raise AssertionError("el audio para ASR no existe")
        on_progress(1.0)
        segment = Segment(id=0, start=0.0, end=1.0, text="hola", avg_logprob=0.0, no_speech_prob=0.0)
        return Transcript(
            engine="fake",
            model="fake",
            device="cpu",
            compute_type="int8",
            language="es",
            language_probability=1.0,
            duration_s=1.0,
            source=source_label,
            segments=(segment,),
            notice="transcripción de prueba",
        )


@pytest.fixture
def fake_vision() -> FakeVisionDetector:
    return FakeVisionDetector()


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return Vault.open_or_create(tmp_path / "vault", TEST_PASSPHRASE, kdf_cost=FAST_KDF, chunk_size=TEST_CHUNK_SIZE)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def tools(settings: Settings) -> FfmpegTools:
    try:
        return FfmpegTools.resolve(settings.ffmpeg_bin, settings.ffprobe_bin, timeout_s=GENERATION_TIMEOUT_SECONDS)
    except MediaError:
        pytest.skip("ffmpeg/ffprobe no están instalados")


@pytest.fixture(scope="session")
def rnnoise_dir(settings: Settings) -> Path:
    if not settings.rnnoise_path.is_file():
        pytest.skip("modelo RNNoise no aprovisionado")
    return settings.rnnoise_dir


def _generate(ffmpeg: str, args: list[str]) -> None:
    """Genera medios sintéticos con lavfi (solo en tests; nunca evidencia real)."""
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        check=True,
        capture_output=True,
        timeout=GENERATION_TIMEOUT_SECONDS,
    )


@pytest.fixture(scope="session")
def media_samples(tmp_path_factory: pytest.TempPathFactory, tools: FfmpegTools) -> MediaSamples:
    root = tmp_path_factory.mktemp("media")
    samples = MediaSamples(root / "tono_con_ruido.wav", root / "clip.mp4", root / "mudo.mp4", root / "voz.webm")
    _generate(
        tools.ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:amplitude=0.2:duration=3",
            "-filter_complex",
            "amix=inputs=2:duration=shortest",
            str(samples.noisy_audio),
        ],
    )
    _generate(
        tools.ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=15:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(samples.video),
        ],
    )
    _generate(
        tools.ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=15:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(samples.silent_video),
        ],
    )
    _generate(
        tools.ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=2",
            "-c:a",
            "libopus",
            str(samples.webm_audio),
        ],
    )
    return samples


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Voz sintética offline con el motor SAPI de Windows (para probar Whisper sin datos reales)."""
    powershell = shutil.which("powershell")
    if sys.platform != "win32" or powershell is None:
        pytest.skip("requiere la síntesis de voz SAPI de Windows")
    target = tmp_path_factory.mktemp("speech") / "speech.wav"
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{target}'); $s.Speak('{SPEECH_TEXT}'); $s.Dispose()"
    )
    subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        timeout=GENERATION_TIMEOUT_SECONDS,
    )
    return target

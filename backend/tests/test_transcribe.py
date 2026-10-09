"""Transcripción local con faster-whisper (CPU, int8)."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.media import FfmpegTools, extract_asr_audio
from app.transcribe import (
    ModelNotAvailableError,
    TranscriptionError,
    WhisperTranscriber,
    load_pcm16_mono,
    to_segment,
)


def _write_wav(path: Path, *, channels: int, rate: int, samples: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return path


def _transcriber(models_root: Path, model_name: str) -> WhisperTranscriber:
    return WhisperTranscriber(models_root=models_root, model_name=model_name, language=None, cpu_threads=0, beam_size=1)


def test_missing_model_is_reported(tmp_path: Path) -> None:
    transcriber = _transcriber(tmp_path, "inexistente")
    assert not transcriber.is_available()
    with pytest.raises(ModelNotAvailableError):
        transcriber.transcribe(tmp_path / "a.wav", source_label="prueba", on_progress=lambda _: None)


def test_segment_conversion_keeps_word_timestamps() -> None:
    raw = SimpleNamespace(
        id=1,
        start=0.123456,
        end=2.5,
        text="  hola mundo ",
        avg_logprob=-0.21,
        no_speech_prob=0.01,
        words=[SimpleNamespace(start=0.12, end=0.5, word=" hola", probability=0.98)],
    )
    segment = to_segment(raw)
    assert segment.text == "hola mundo"
    assert segment.start == 0.123
    assert segment.words[0].text == " hola"
    assert segment.words[0].probability == 0.98


def test_pcm_loader_scales_samples_to_unit_range(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "a.wav", channels=1, rate=16000, samples=np.array([0, 16384, -32768]))
    audio = load_pcm16_mono(path)
    assert audio.dtype == np.float32
    assert audio.tolist() == [0.0, 0.5, -1.0]


@pytest.mark.parametrize(("channels", "rate"), [(2, 16000), (1, 44100)])
def test_pcm_loader_rejects_unexpected_formats(tmp_path: Path, channels: int, rate: int) -> None:
    path = _write_wav(tmp_path / "a.wav", channels=channels, rate=rate, samples=np.zeros(8))
    with pytest.raises(TranscriptionError):
        load_pcm16_mono(path)


@pytest.mark.slow
@pytest.mark.ffmpeg
def test_tiny_model_transcribes_synthetic_speech(
    settings: Settings, tools: FfmpegTools, speech_wav: Path, tmp_path: Path
) -> None:
    transcriber = _transcriber(settings.models_dir, "tiny")
    if not transcriber.is_available():
        pytest.skip("modelo whisper tiny no aprovisionado")
    asr_audio = extract_asr_audio(tools, speech_wav, tmp_path / "asr.wav", audio_filter=None, cwd=None).output
    progress: list[float] = []

    transcript = transcriber.transcribe(asr_audio, source_label="sintetico", on_progress=progress.append)

    text = " ".join(s.text for s in transcript.segments).lower()
    assert "test" in text
    assert transcript.language == "en"
    assert (transcript.device, transcript.compute_type) == ("cpu", "int8")
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert all(w.start <= w.end for s in transcript.segments for w in s.words)
    assert progress and progress[-1] <= 1.0

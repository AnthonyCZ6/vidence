"""Muestreo de fotogramas con OpenCV."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.frames import FrameExtractionError, effective_interval, iter_frames
from tests.conftest import MediaSamples


@pytest.mark.parametrize(
    ("requested", "duration", "max_frames", "expected"),
    [(5.0, 60.0, 100, 5.0), (1.0, 1000.0, 100, 10.0), (5.0, None, 100, 5.0), (5.0, 0.0, 100, 5.0)],
)
def test_effective_interval(requested: float, duration: float | None, max_frames: int, expected: float) -> None:
    assert effective_interval(requested, duration, max_frames) == expected


@pytest.mark.ffmpeg
def test_frames_are_sampled_once_per_interval(media_samples: MediaSamples) -> None:
    frames = list(iter_frames(media_samples.video, interval_s=1.0, max_frames=100))

    assert len(frames) == 4
    assert [f.index for f in frames] == [0, 1, 2, 3]
    for frame in frames:
        assert frame.timestamp_s >= frame.index * 1.0 - 0.01
        assert frame.timestamp_s < (frame.index + 1) * 1.0
        assert frame.jpeg.startswith(b"\xff\xd8")
        assert (frame.width, frame.height) == (320, 240)


@pytest.mark.ffmpeg
def test_frame_count_is_capped(media_samples: MediaSamples) -> None:
    assert len(list(iter_frames(media_samples.video, interval_s=0.25, max_frames=3))) == 3


def test_unreadable_video_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "roto.mp4"
    bogus.write_bytes(b"esto no es un video")
    with pytest.raises(FrameExtractionError):
        list(iter_frames(bogus, interval_s=1.0, max_frames=10))

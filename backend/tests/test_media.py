"""ffprobe/ffmpeg: validación de formatos, cadenas de filtros y renderizado."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from app import media
from app.media import (
    ENHANCE_VIDEO_FILTER,
    FfmpegError,
    FfmpegTools,
    LoudnormMeasurement,
    MediaError,
    UnsupportedMediaError,
    build_audio_filter,
    build_render_args,
    extract_asr_audio,
    is_crash_exit,
    measure_loudness,
    parse_loudnorm_output,
    parse_probe,
    portable_arguments,
    probe,
    render,
)
from tests.conftest import MediaSamples

LOUDNORM_STDERR = """
[Parsed_loudnorm_2 @ 000001]
{
	"input_i" : "-27.61",
	"input_tp" : "-4.47",
	"input_lra" : "18.06",
	"input_thresh" : "-39.20",
	"output_i" : "-16.58",
	"output_tp" : "-1.50",
	"output_lra" : "14.78",
	"output_thresh" : "-27.71",
	"normalization_type" : "dynamic",
	"target_offset" : "0.58"
}
"""


def _probe_payload(format_name: str, streams: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format": {
            "format_name": format_name,
            "duration": "4.000000",
            "size": "1000",
            "bit_rate": "2000",
            "tags": {"creation_time": "2026-10-08T10:00:00.000000Z"},
        },
        "streams": streams,
    }


VIDEO_STREAM = {
    "index": 0,
    "codec_type": "video",
    "codec_name": "h264",
    "width": 320,
    "height": 240,
    "avg_frame_rate": "15/1",
}
AUDIO_STREAM = {"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 2}


# --- Análisis de ffprobe (puro) ---------------------------------------------------


def test_parse_probe_detects_audio_video_and_metadata() -> None:
    info = parse_probe(_probe_payload("mov,mp4,m4a,3gp,3g2,mj2", [VIDEO_STREAM, AUDIO_STREAM]))
    assert info.has_video and info.has_audio
    assert info.duration_s == 4.0
    assert info.creation_time == "2026-10-08T10:00:00.000000Z"
    assert info.streams[0].frame_rate == 15.0
    assert info.streams[1].sample_rate == 44100


def test_cover_art_is_not_treated_as_video() -> None:
    cover = {**VIDEO_STREAM, "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}
    info = parse_probe(_probe_payload("mp3", [cover, AUDIO_STREAM]))
    assert info.has_audio and not info.has_video


@pytest.mark.parametrize("format_name", ["hls", "concat", "image2", ""])
def test_disallowed_containers_are_rejected(format_name: str) -> None:
    with pytest.raises(UnsupportedMediaError):
        parse_probe(_probe_payload(format_name, [AUDIO_STREAM]))


def test_file_without_audio_or_video_is_rejected() -> None:
    subtitles = {"index": 0, "codec_type": "subtitle", "codec_name": "mov_text"}
    with pytest.raises(UnsupportedMediaError):
        parse_probe(_probe_payload("mov,mp4,m4a,3gp,3g2,mj2", [subtitles]))


def test_unparseable_numbers_become_none() -> None:
    payload = _probe_payload("wav", [AUDIO_STREAM])
    payload["format"] = {**payload["format"], "duration": "N/A"}  # type: ignore[dict-item]
    assert parse_probe(payload).duration_s is None


# --- Cadenas de filtros (puro) ------------------------------------------------------


def test_audio_filter_uses_rnnoise_when_model_available() -> None:
    chain = build_audio_filter("cb.rnnn", None)
    assert chain.startswith("highpass=f=80,arnndn=m=cb.rnnn,aeval=")
    assert "loudnorm=" in chain
    assert chain.endswith("aresample=48000")


def test_rnnoise_output_is_sanitized_before_loudnorm() -> None:
    chain = build_audio_filter("cb.rnnn", None)
    assert chain.index("isnan") < chain.index("loudnorm")
    assert "clip(val(ch),-1,1)" in chain


def test_rnnoise_tail_is_padded_then_trimmed_to_original_duration() -> None:
    chain = build_audio_filter("cb.rnnn", None, duration_s=16.981333)
    assert "apad=pad_dur=0.1,arnndn=m=cb.rnnn," in chain
    assert "atrim=duration=16.981333" in chain
    assert chain.index("atrim") < chain.index("loudnorm")


def test_rnnoise_without_known_duration_is_not_padded() -> None:
    chain = build_audio_filter("cb.rnnn", None)
    assert "apad" not in chain
    assert "atrim" not in chain


def test_audio_duration_prefers_audio_stream() -> None:
    with_stream_duration = parse_probe(_probe_payload("wav", [{**AUDIO_STREAM, "duration": "3.5"}]))
    without_stream_duration = parse_probe(_probe_payload("wav", [AUDIO_STREAM]))
    assert with_stream_duration.audio_duration_s == 3.5
    assert without_stream_duration.audio_duration_s == 4.0


def test_audio_filter_falls_back_to_fft_denoiser() -> None:
    assert "afftdn" in build_audio_filter(None, None)


def test_audio_filter_applies_measured_loudness_in_second_pass() -> None:
    measurement = LoudnormMeasurement(-27.61, -4.47, 18.06, -39.2, 0.58)
    chain = build_audio_filter("cb.rnnn", measurement)
    assert "measured_I=-27.61" in chain
    assert "offset=0.58" in chain
    assert "linear=true" in chain


@pytest.mark.parametrize("bad_model", ["../x.rnnn", "a,b", "m'x", "a:b"])
def test_audio_filter_rejects_unsafe_model_names(bad_model: str) -> None:
    with pytest.raises(ValueError):
        build_audio_filter(bad_model, None)


def test_parse_loudnorm_output_reads_measurement() -> None:
    assert parse_loudnorm_output(LOUDNORM_STDERR) == LoudnormMeasurement(-27.61, -4.47, 18.06, -39.2, 0.58)


def test_parse_loudnorm_output_returns_none_for_silence() -> None:
    assert parse_loudnorm_output(LOUDNORM_STDERR.replace('"-27.61"', '"-inf"')) is None


def test_parse_loudnorm_output_rejects_impossible_measurements() -> None:
    """Un pico de basura de arnndn llegó a medir +29.95 LUFS; loudnorm lo rechazaría (fuera de [-99, 0])."""
    assert parse_loudnorm_output(LOUDNORM_STDERR.replace('"-27.61"', '"29.95"')) is None


def test_parse_loudnorm_output_returns_none_without_json() -> None:
    assert parse_loudnorm_output("sin datos") is None


# --- Argumentos de ffmpeg (puro) -------------------------------------------------------


def test_render_args_restrict_protocols_and_demuxers_before_input() -> None:
    args = build_render_args(
        "ffmpeg",
        Path("in.mp4"),
        Path("out.mp4"),
        has_video=True,
        has_audio=True,
        video_filter=ENHANCE_VIDEO_FILTER,
        audio_filter="anull",
    )
    before_input = args[: args.index("-i")]
    assert before_input[before_input.index("-protocol_whitelist") + 1] == "file"
    assert "-format_whitelist" in before_input
    assert "-nostdin" in args
    assert args[-1] == "out.mp4"


def test_audio_only_render_uses_flac_and_drops_video() -> None:
    args = build_render_args(
        "ffmpeg",
        Path("in.wav"),
        Path("out.flac"),
        has_video=False,
        has_audio=True,
        video_filter=None,
        audio_filter=None,
    )
    assert "-vn" in args
    assert args[args.index("-c:a") + 1] == "flac"


def test_portable_arguments_hide_local_paths() -> None:
    src, dst = Path("C:/datos/ws/original.mp4"), Path("C:/datos/ws/enhanced.mp4")
    args = ["C:/ffmpeg/bin/ffmpeg.exe", "-i", str(src), "-c:v", "libx264", str(dst)]
    assert portable_arguments(args, src, dst) == ("ffmpeg", "-i", "{input}", "-c:v", "libx264", "{output}")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0xC0000005, True),
        (0xC0000409, True),
        (-11, True),
        (0, False),
        (1, False),
        (183, False),
        (0xFFFFFFDE, False),  # AVERROR(ERANGE) = -34 visto como código sin signo en Windows
    ],
)
def test_crash_exit_codes_are_recognized(code: int, expected: bool) -> None:
    assert is_crash_exit(code) is expected


def _fake_subprocess(monkeypatch: pytest.MonkeyPatch, codes: list[int]) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return subprocess.CompletedProcess(args, codes[min(len(calls), len(codes)) - 1], b"", b"detalle")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    return calls


def test_crashed_ffmpeg_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_subprocess(monkeypatch, [0xC0000005, 0])
    media._run(["ffmpeg", "-version"], timeout_s=5)
    assert len(calls) == 2


def test_repeated_crash_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_subprocess(monkeypatch, [0xC0000005])
    with pytest.raises(FfmpegError):
        media._run(["ffmpeg", "-version"], timeout_s=5)
    assert len(calls) == 2


def test_ordinary_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_subprocess(monkeypatch, [1])
    with pytest.raises(FfmpegError):
        media._run(["ffmpeg", "-version"], timeout_s=5)
    assert len(calls) == 1


def test_missing_binaries_raise_media_error() -> None:
    with pytest.raises(MediaError):
        FfmpegTools.resolve("no-existe-ffmpeg", "no-existe-ffprobe", timeout_s=10)


# --- Integración con ffmpeg real ----------------------------------------------------------


@pytest.mark.ffmpeg
def test_tools_report_ffmpeg_version(tools: FfmpegTools) -> None:
    assert tools.version.startswith("ffmpeg version")


@pytest.mark.ffmpeg
def test_probe_reads_generated_video(tools: FfmpegTools, media_samples: MediaSamples) -> None:
    info = probe(tools, media_samples.video)
    assert info.has_video and info.has_audio
    assert info.duration_s == pytest.approx(4.0, abs=0.2)


@pytest.mark.ffmpeg
def test_probe_accepts_browser_recording_format(tools: FfmpegTools, media_samples: MediaSamples) -> None:
    info = probe(tools, media_samples.webm_audio)
    assert info.has_audio and not info.has_video


@pytest.mark.ffmpeg
@pytest.mark.parametrize("suffix", [".m3u8", ".mp4"])
def test_probe_rejects_hls_playlist_that_points_to_local_files(
    tools: FfmpegTools, media_samples: MediaSamples, tmp_path: Path, suffix: str
) -> None:
    playlist = tmp_path / f"trampa{suffix}"
    playlist.write_text(
        f"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:3.0,\nfile:{media_samples.noisy_audio.as_posix()}\n#EXT-X-ENDLIST\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedMediaError):
        probe(tools, playlist)


@pytest.mark.ffmpeg
def test_enhanced_audio_is_48k_flac(
    tools: FfmpegTools, media_samples: MediaSamples, rnnoise_dir: Path, tmp_path: Path
) -> None:
    media = probe(tools, media_samples.noisy_audio)
    duration = media.audio_duration_s
    measurement = measure_loudness(
        tools, media_samples.noisy_audio, rnnoise_model="cb.rnnn", duration_s=duration, cwd=rnnoise_dir
    )
    assert measurement is not None

    result = render(
        tools,
        media_samples.noisy_audio,
        tmp_path / "enhanced.flac",
        media=media,
        video_filter=None,
        audio_filter=build_audio_filter("cb.rnnn", measurement, duration),
        cwd=rnnoise_dir,
    )

    out = probe(tools, result.output)
    assert out.format_name == "flac"
    assert out.streams[0].sample_rate == 48000
    assert out.duration_s == pytest.approx(duration, abs=0.03)
    assert "{input}" in result.arguments and "{output}" in result.arguments


@pytest.mark.ffmpeg
def test_rnnoise_chain_never_emits_nan_samples(
    tools: FfmpegTools, media_samples: MediaSamples, rnnoise_dir: Path
) -> None:
    """arnndn (ffmpeg 9.0.2) llena la última trama con memoria sin inicializar (NaN o picos de +290 dB).

    Sin la corrección, ~50 % de las ejecuciones la producen: 6 repeticiones lo detectan con ~98 %.
    astats solo cuenta NaN en formato de coma flotante, de ahí el aformat previo.
    """
    media_info = probe(tools, media_samples.noisy_audio)
    chain = build_audio_filter("cb.rnnn", None, duration_s=media_info.audio_duration_s)
    args = [
        tools.ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(media_samples.noisy_audio),
        "-af",
        f"{chain},aformat=sample_fmts=flt,astats=measure_perchannel=none",
        "-f",
        "null",
        "-",
    ]
    for _ in range(6):
        stderr = subprocess.run(args, cwd=rnnoise_dir, capture_output=True, check=True, timeout=60).stderr
        report = stderr.decode("utf-8", errors="replace")
        assert "Number of NaNs: 0" in report
        peak = re.findall(r"Peak level dB: (\S+)", report)[-1]
        assert float(peak) <= 0.0


@pytest.mark.ffmpeg
def test_enhanced_video_is_browser_playable_mp4(
    tools: FfmpegTools, media_samples: MediaSamples, tmp_path: Path
) -> None:
    media = probe(tools, media_samples.video)
    result = render(
        tools,
        media_samples.video,
        tmp_path / "enhanced.mp4",
        media=media,
        video_filter=ENHANCE_VIDEO_FILTER,
        audio_filter=build_audio_filter(None, None),
        cwd=None,
    )
    out = probe(tools, result.output)
    codecs = {s.codec_type: s.codec_name for s in out.streams}
    assert codecs == {"video": "h264", "audio": "aac"}


@pytest.mark.ffmpeg
def test_video_without_audio_renders(tools: FfmpegTools, media_samples: MediaSamples, tmp_path: Path) -> None:
    media = probe(tools, media_samples.silent_video)
    result = render(
        tools,
        media_samples.silent_video,
        tmp_path / "enhanced.mp4",
        media=media,
        video_filter=ENHANCE_VIDEO_FILTER,
        audio_filter=None,
        cwd=None,
    )
    assert not probe(tools, result.output).has_audio


@pytest.mark.ffmpeg
def test_asr_audio_is_16k_mono(tools: FfmpegTools, media_samples: MediaSamples, tmp_path: Path) -> None:
    extract_asr_audio(tools, media_samples.video, tmp_path / "asr.wav", audio_filter=None, cwd=None)
    stream = probe(tools, tmp_path / "asr.wav").streams[0]
    assert (stream.sample_rate, stream.channels) == (16000, 1)


@pytest.mark.ffmpeg
def test_ffmpeg_failure_raises_ffmpeg_error(tools: FfmpegTools, tmp_path: Path) -> None:
    with pytest.raises(FfmpegError):
        extract_asr_audio(tools, tmp_path / "no-existe.wav", tmp_path / "asr.wav", audio_filter=None, cwd=None)

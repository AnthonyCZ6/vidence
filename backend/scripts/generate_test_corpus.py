"""Genera un corpus sintético de pruebas (videos y audios), 100 % offline.

Voz: síntesis SAPI de Windows (voces en inglés instaladas). Imagen y ruido:
fuentes lavfi de ffmpeg. Nunca se usa evidencia real.

Escribe <salida>/corpus.json con lo que se espera de cada archivo; lo consume
scripts/e2e_stress.py.

Uso:
    python scripts/generate_test_corpus.py                 # -> samples/corpus/
    python scripts/generate_test_corpus.py --out C:/ruta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PROJECT_ROOT, Settings

logger = logging.getLogger("corpus")

DEFAULT_OUT = PROJECT_ROOT / "samples" / "corpus"
TIMEOUT_S = 600
TRUNCATE_FRACTION = 0.4

SPEECH = {
    "s1": "This is test number one. The quick brown fox jumps over the lazy dog near the river.",
    "s2": "This recording was made on a mobile phone. Please check every timestamp carefully.",
    "s3": "The meeting started at nine in the morning and finished before noon on Friday.",
}
REFERENCE = SPEECH  # texto pronunciado en cada archivo: referencia para calcular el WER
H264 = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p")


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    description: str
    outcome: str  # processed | processing_fails | upload_rejected
    has_video: bool = False
    has_audio: bool = False
    duration_s: float | None = None
    transcript: str = "none"  # speech | speech_soft | silence | none
    reference: str = ""
    absent_keywords: tuple[str, ...] = ()
    portrait: bool = False
    expected_status: int | None = None
    expected_error: str | None = None
    sha256: str = ""
    size_bytes: int = 0


def video_src(width: int, height: int, rate: int, duration: float) -> list[str]:
    return ["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={rate}:duration={duration}"]


def noise_src(amplitude: float, duration: float) -> list[str]:
    return ["-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={amplitude}:duration={duration}:sample_rate=48000"]


def speech_mix(speech_input: int, noise_input: int) -> str:
    """Voz rellenada con silencio y mezclada con ruido rosa; dura lo que dure el ruido."""
    return (
        f"[{speech_input}:a]aresample=48000,apad[s];[s][{noise_input}:a]amix=inputs=2:duration=shortest:normalize=0[a]"
    )


class CorpusBuilder:
    def __init__(self, ffmpeg: str, out: Path) -> None:
        self._ffmpeg = ffmpeg
        self._out = out
        self._sources = out / "_fuentes"
        self._sources.mkdir(parents=True, exist_ok=True)
        self._speech = self._synthesize_speech()

    def _path(self, name: str) -> Path:
        return self._out / name

    def _run(self, *args: str | Path) -> None:
        command = [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)]
        result = subprocess.run(command, capture_output=True, timeout=TIMEOUT_S, check=False)  # noqa: S603
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falló: {result.stderr.decode('utf-8', 'replace')[-800:]}")

    def _synthesize_speech(self) -> dict[str, Path]:
        powershell = shutil.which("powershell")
        if sys.platform != "win32" or powershell is None:
            raise RuntimeError("Se requiere la síntesis de voz SAPI de Windows")
        paths = {key: self._sources / f"{key}.wav" for key in SPEECH}
        lines = ["Add-Type -AssemblyName System.Speech", "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer"]
        lines.append("$s.Rate = -1")
        for key, text in SPEECH.items():
            lines += [f"$s.SetOutputToWaveFile('{paths[key]}')", f"$s.Speak('{text}')"]
        lines.append("$s.Dispose()")
        subprocess.run(  # noqa: S603
            [powershell, "-NoProfile", "-NonInteractive", "-Command", "; ".join(lines)],
            check=True,
            capture_output=True,
            timeout=TIMEOUT_S,
        )
        return paths

    # --- Casos que deben procesarse ------------------------------------------------

    def h264_720p(self) -> CorpusEntry:
        name = "01_h264_aac_720p30.mp4"
        self._run(
            *video_src(1280, 720, 30, 20),
            "-i",
            self._speech["s1"],
            *noise_src(0.1, 20),
            "-filter_complex",
            speech_mix(1, 2),
            "-map",
            "0:v",
            "-map",
            "[a]",
            *H264,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            self._path(name),
        )  # sin faststart: moov al final
        return CorpusEntry(
            name, "H.264 + AAC 720p30, voz con ruido rosa", "processed", True, True, 20.0, "speech", REFERENCE["s1"]
        )

    def phone_portrait_vfr(self) -> CorpusEntry:
        name = "02_telefono_vertical_vfr.mp4"
        temporary = self._sources / "02_sin_rotacion.mp4"
        self._run(
            *video_src(640, 360, 30, 8),
            *video_src(640, 360, 15, 8),
            "-i",
            self._speech["s2"],
            *noise_src(0.08, 16),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v];" + speech_mix(2, 3),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-fps_mode",
            "vfr",
            *H264,
            "-c:a",
            "aac",
            temporary,
        )
        self._run("-display_rotation", "90", "-i", temporary, "-c", "copy", self._path(name))
        return CorpusEntry(
            name,
            "Teléfono: vertical (rotación 90°) y frame rate variable 30/15 fps",
            "processed",
            True,
            True,
            16.0,
            "speech",
            REFERENCE["s2"],
            portrait=True,
        )

    def hevc_mov(self) -> CorpusEntry:
        name = "03_hevc_aac_1080p.mov"
        self._run(
            *video_src(1920, 1080, 30, 10),
            "-i",
            self._speech["s3"],
            *noise_src(0.08, 10),
            "-filter_complex",
            speech_mix(1, 2),
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            self._path(name),
        )
        return CorpusEntry(
            name, "HEVC/H.265 1080p en MOV (estilo iPhone)", "processed", True, True, 10.0, "speech", REFERENCE["s3"]
        )

    def vp9_webm_live(self) -> CorpusEntry:
        name = "04_vp9_opus_mediarecorder.webm"
        self._run(
            *video_src(640, 480, 24, 12),
            "-i",
            self._speech["s1"],
            *noise_src(0.1, 12),
            "-filter_complex",
            speech_mix(1, 2),
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-b:v",
            "600k",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            "-live",
            "1",
            self._path(name),
        )
        return CorpusEntry(
            name,
            "VP9 + Opus en WebM sin duración en la cabecera (como MediaRecorder)",
            "processed",
            True,
            True,
            12.0,
            "speech",
            REFERENCE["s1"],
        )

    def odd_dims_mkv(self) -> CorpusEntry:
        name = "05_dimensiones_impares_mp3.mkv"
        # testsrc2 redondea a dimensiones pares: el tamaño impar se fuerza con scale (yuv444p lo admite).
        self._run(
            *video_src(640, 360, 25, 10),
            "-i",
            self._speech["s2"],
            *noise_src(0.08, 10),
            "-filter_complex",
            "[0:v]scale=641:361,format=yuv444p[v];" + speech_mix(1, 2),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            self._path(name),
        )
        return CorpusEntry(
            name, "MKV 641x361 (impar) yuv444p con audio MP3", "processed", True, True, 10.0, "speech", REFERENCE["s2"]
        )

    def silent_video(self) -> CorpusEntry:
        name = "06_video_sin_audio.mp4"
        self._run(*video_src(640, 360, 24, 10), *H264, "-an", self._path(name))
        return CorpusEntry(name, "Video sin pista de audio", "processed", True, False, 10.0, "none")

    def silence_track(self) -> CorpusEntry:
        name = "07_pista_de_silencio.mp4"
        self._run(
            *video_src(640, 360, 24, 10),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            "10",
            "-map",
            "0:v",
            "-map",
            "1:a",
            *H264,
            "-c:a",
            "aac",
            self._path(name),
        )
        return CorpusEntry(
            name, "Video con pista de audio en silencio absoluto", "processed", True, True, 10.0, "silence"
        )

    def m4a_speech(self) -> CorpusEntry:
        name = "08_solo_audio.m4a"
        self._run(
            "-i",
            self._speech["s3"],
            *noise_src(0.08, 10),
            "-filter_complex",
            speech_mix(0, 1),
            "-map",
            "[a]",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            self._path(name),
        )
        return CorpusEntry(name, "Solo audio AAC (M4A)", "processed", False, True, 10.0, "speech", REFERENCE["s3"])

    def phone_8k_wav(self) -> CorpusEntry:
        name = "09_telefonico_8khz.wav"
        self._run(
            "-i",
            self._speech["s2"],
            *noise_src(0.05, 10),
            "-filter_complex",
            speech_mix(0, 1),
            "-map",
            "[a]",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            self._path(name),
        )
        return CorpusEntry(
            name, "Calidad telefónica: WAV 8 kHz mono", "processed", False, True, 10.0, "speech", REFERENCE["s2"]
        )

    def mp3_cover(self) -> CorpusEntry:
        name = "10_mp3_con_caratula.mp3"
        cover = self._sources / "caratula.jpg"
        self._run("-f", "lavfi", "-i", "color=c=0x0f766e:size=300x300", "-frames:v", "1", cover)
        # La carátula entra como imagen aparte: con -frames:v en la salida, ffmpeg 9 cortaría también el audio.
        self._run(
            "-i",
            self._speech["s1"],
            *noise_src(0.08, 10),
            "-i",
            cover,
            "-filter_complex",
            speech_mix(0, 1),
            "-map",
            "[a]",
            "-map",
            "2:v",
            "-c:v",
            "copy",
            "-disposition:v:0",
            "attached_pic",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-id3v2_version",
            "3",
            self._path(name),
        )
        return CorpusEntry(
            name,
            "MP3 estéreo 44.1 kHz con carátula (no debe tratarse como video)",
            "processed",
            False,
            True,
            10.0,
            "speech",
            REFERENCE["s1"],
        )

    def ogg_vorbis(self) -> CorpusEntry:
        name = "11_vorbis.ogg"
        self._run(
            "-i",
            self._speech["s1"],
            *noise_src(0.08, 10),
            "-filter_complex",
            speech_mix(0, 1),
            "-map",
            "[a]",
            "-c:a",
            "libvorbis",
            "-q:a",
            "4",
            self._path(name),
        )
        return CorpusEntry(name, "Ogg Vorbis", "processed", False, True, 10.0, "speech", REFERENCE["s1"])

    def high_noise_flac(self) -> CorpusEntry:
        name = "12_ruido_intenso.flac"
        self._run(
            "-i",
            self._speech["s3"],
            *noise_src(0.35, 10),
            "-filter_complex",
            speech_mix(0, 1),
            "-map",
            "[a]",
            "-c:a",
            "flac",
            self._path(name),
        )
        return CorpusEntry(
            name,
            "FLAC con ruido intenso (relación señal/ruido baja)",
            "processed",
            False,
            True,
            10.0,
            "speech_soft",
            REFERENCE["s3"],
        )

    def long_video(self) -> CorpusEntry:
        name = "13_video_largo_90s.mp4"
        speech = (
            "[1:a]aresample=48000,apad=pad_dur=4[a1];[2:a]aresample=48000,apad=pad_dur=4[a2];"
            "[3:a]aresample=48000[a3];[a1][a2][a3]concat=n=3:v=0:a=1,apad[s];"
            "[s][4:a]amix=inputs=2:duration=shortest:normalize=0[a]"
        )
        self._run(
            *video_src(640, 360, 24, 90),
            "-i",
            self._speech["s1"],
            "-i",
            self._speech["s2"],
            "-i",
            self._speech["s3"],
            *noise_src(0.08, 90),
            "-filter_complex",
            speech,
            "-map",
            "0:v",
            "-map",
            "[a]",
            *H264,
            "-c:a",
            "aac",
            self._path(name),
        )
        return CorpusEntry(
            name,
            "Video de 90 s: tres frases y un tramo largo solo con ruido",
            "processed",
            True,
            True,
            90.0,
            "speech",
            " ".join(SPEECH.values()),
        )

    def multi_audio_mkv(self) -> CorpusEntry:
        name = "14_dos_pistas_de_audio.mkv"
        self._run(
            *video_src(640, 360, 24, 10),
            "-i",
            self._speech["s1"],
            *noise_src(0.08, 10),
            "-i",
            self._speech["s2"],
            "-filter_complex",
            speech_mix(1, 2) + ";[3:a]aresample=48000,apad=whole_dur=10[b]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-map",
            "[b]",
            *H264,
            "-c:a",
            "aac",
            self._path(name),
        )
        return CorpusEntry(
            name,
            "MKV con dos pistas de audio (se procesa la primera)",
            "processed",
            True,
            True,
            10.0,
            "speech",
            REFERENCE["s1"],
            absent_keywords=("mobile",),
        )

    # --- Casos que deben rechazarse -------------------------------------------------------

    def truncated_mp4(self, source: CorpusEntry) -> CorpusEntry:
        name = "90_truncado.mp4"
        data = self._path(source.name).read_bytes()
        self._path(name).write_bytes(data[: int(len(data) * TRUNCATE_FRACTION)])
        return CorpusEntry(
            name, "MP4 cortado al 40 % (sin índice moov)", "processing_fails", expected_error="Formato no compatible"
        )

    def text_disguised(self) -> CorpusEntry:
        name = "91_texto_disfrazado.mp4"
        self._path(name).write_text("esto no es un video\n" * 200, encoding="utf-8")
        return CorpusEntry(name, "Texto con extensión .mp4", "processing_fails", expected_error="Formato no compatible")

    def hls_trap(self, target: CorpusEntry) -> CorpusEntry:
        name = "92_trampa_hls.mp4"
        playlist = f"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:10.0,\nfile:{self._path(target.name).as_posix()}\n"
        self._path(name).write_text(playlist + "#EXT-X-ENDLIST\n", encoding="utf-8")
        return CorpusEntry(
            name,
            "Playlist HLS disfrazada que intenta leer otro archivo local",
            "processing_fails",
            expected_error="Formato no compatible",
        )

    def executable(self) -> CorpusEntry:
        name = "93_programa.exe"
        self._path(name).write_bytes(b"MZ" + os.urandom(4096))
        return CorpusEntry(name, "Ejecutable: debe rechazarse en la subida", "upload_rejected", expected_status=415)

    def build(self) -> list[CorpusEntry]:
        processed = [
            self.h264_720p(),
            self.phone_portrait_vfr(),
            self.hevc_mov(),
            self.vp9_webm_live(),
            self.odd_dims_mkv(),
            self.silent_video(),
            self.silence_track(),
            self.m4a_speech(),
            self.phone_8k_wav(),
            self.mp3_cover(),
            self.ogg_vorbis(),
            self.high_noise_flac(),
            self.long_video(),
            self.multi_audio_mkv(),
        ]
        rejected = [
            self.truncated_mp4(processed[0]),
            self.text_disguised(),
            self.hls_trap(processed[7]),
            self.executable(),
        ]
        return [self._with_digest(entry) for entry in processed + rejected]

    def _with_digest(self, entry: CorpusEntry) -> CorpusEntry:
        data = self._path(entry.name).read_bytes()
        return replace(entry, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="carpeta de salida")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    entries = CorpusBuilder(Settings().ffmpeg_bin, args.out).build()
    manifest = args.out / "corpus.json"
    manifest.write_text(json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in entries:
        logger.info("%-34s %9d bytes  %s", entry.name, entry.size_bytes, entry.description)
    logger.info("Corpus de %d archivos en %s", len(entries), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

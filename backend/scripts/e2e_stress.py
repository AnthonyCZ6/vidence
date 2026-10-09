"""Prueba de estrés de punta a punta contra un servidor Vidence real.

Arranca `python -m app` con una bóveda temporal desechable, sube N veces cada
archivo del corpus (scripts/generate_test_corpus.py), verifica cada resultado
y escribe un reporte en reports/. Todo ocurre en loopback.

Uso:
    python scripts/e2e_stress.py --iterations 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools

import httpx

from app.config import BACKEND_ROOT, PROJECT_ROOT, Settings
from app.media import FfmpegTools, probe

logger = logging.getLogger("e2e")

CSRF = {"X-Requested-With": "vidence"}
PASSPHRASE = "frase-desechable-de-estres-e2e"  # noqa: S105  (bóveda temporal creada solo para esta prueba)
JOB_TIMEOUT_S = 900
HEALTH_TIMEOUT_S = 90
POLL_INTERVAL_S = 0.5
DURATION_TOLERANCE_S = 0.15
PEAK_LIMIT_DB = 0.5  # los picos de basura de arnndn llegaban a +38..+291 dB
RANGE_PROBE = (100, 1099)
PLAINTEXT_MARKERS = (b"brown fox", b"mobile phone", b"meeting started", b"h264_aac_720p30", b"telefono_vertical")
WER_LIMIT = 0.25
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}  # fmt: skip


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class TranscriptVerdict:
    checks: tuple[Check, ...]
    soft_misses: tuple[str, ...] = ()
    wer: float | None = None


@dataclass(frozen=True)
class CaseResult:
    iteration: int
    name: str
    seconds: float
    checks: tuple[Check, ...]
    soft_misses: tuple[str, ...] = field(default=())
    wer: float | None = None

    @property
    def passed(self) -> bool:
        return all(check.ok for check in self.checks)


@dataclass(frozen=True)
class Context:
    client: httpx.Client
    tools: FfmpegTools
    corpus_dir: Path
    data_dir: Path
    scratch: Path
    frame_interval_s: float


# --- Servidor -------------------------------------------------------------------------


@contextmanager
def running_server(data_dir: Path, port: int, log_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    env = {
        **os.environ,
        "VIDENCE_DATA_DIR": str(data_dir),
        "VIDENCE_PORT": str(port),
        "VIDENCE_PASSPHRASE": PASSPHRASE,
        "PYTHONIOENCODING": "utf-8",
    }
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "app"], cwd=BACKEND_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            yield process
        finally:
            process.terminate()
            process.wait(timeout=30)


def wait_until_healthy(client: httpx.Client, process: subprocess.Popen[bytes]) -> dict[str, Any]:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"El servidor terminó al arrancar (código {process.returncode})")
        try:
            return client.get("/api/health").raise_for_status().json()
        except httpx.HTTPError:
            time.sleep(POLL_INTERVAL_S)
    raise RuntimeError("El servidor no respondió a tiempo")


# --- Utilidades HTTP y de medios -----------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def asset_url(evidence_id: str, name: str) -> str:
    return f"/api/evidence/{evidence_id}/assets/{name}"


def wait_for_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_S
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] in ("done", "failed"):
            return job
        time.sleep(POLL_INTERVAL_S)
    return {"stage": "timeout", "error": f"sin terminar tras {JOB_TIMEOUT_S} s"}


def audio_health(tools: FfmpegTools, path: Path) -> tuple[int, float]:
    """(muestras NaN, pico en dBFS) del audio decodificado en coma flotante."""
    args = [
        tools.ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        "aformat=sample_fmts=flt,astats=measure_perchannel=none",
        "-f",
        "null",
        "-",
    ]
    report = subprocess.run(args, capture_output=True, timeout=300, check=True).stderr.decode("utf-8", "replace")  # noqa: S603
    nans = re.findall(r"Number of NaNs: (\S+)", report)
    peaks = re.findall(r"Peak level dB: (\S+)", report)
    return int(float(nans[-1])), float(peaks[-1])


# --- Verificaciones de una evidencia procesada --------------------------------------------------


def verify_derivatives(ctx: Context, entry: dict[str, Any], manifest: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    evidence_id = manifest["evidence_id"]
    expected_type = "video/mp4" if entry["has_video"] else "audio/flac"
    roles = sorted(asset["role"] for asset in manifest["derivatives"])
    checks.append(Check("derivados", roles == ["enhanced", "original_preview"], str(roles)))
    for asset in manifest["derivatives"]:
        content = ctx.client.get(asset_url(evidence_id, asset["name"])).content
        checks.append(Check(f"{asset['role']}: hash", sha256(content) == asset["sha256"]))
        checks.append(Check(f"{asset['role']}: tipo", asset["media_type"] == expected_type, asset["media_type"]))
        checks.append(Check(f"{asset['role']}: linaje", asset["derived_from_sha256"] == entry["sha256"]))
        start, end = RANGE_PROBE
        partial = ctx.client.get(asset_url(evidence_id, asset["name"]), headers={"Range": f"bytes={start}-{end}"})
        checks.append(
            Check(f"{asset['role']}: rango", partial.status_code == 206 and partial.content == content[start : end + 1])
        )
        if asset["role"] == "enhanced":
            checks.extend(verify_enhanced_file(ctx, entry, content, Path(asset["name"]).suffix))
    return checks


def verify_enhanced_file(ctx: Context, entry: dict[str, Any], content: bytes, suffix: str) -> list[Check]:
    path = ctx.scratch / f"enhanced{suffix}"
    path.write_bytes(content)
    info = probe(ctx.tools, path)
    duration = info.duration_s or 0.0
    checks = [
        Check(
            "duración preservada",
            abs(duration - entry["duration_s"]) <= DURATION_TOLERANCE_S,
            f"{duration:.3f}s vs {entry['duration_s']}s",
        )
    ]
    if entry["portrait"]:
        video = next(s for s in info.streams if s.codec_type == "video")
        checks.append(
            Check("orientación vertical", (video.height or 0) > (video.width or 0), f"{video.width}x{video.height}")
        )
    if entry["has_audio"]:
        nans, peak = audio_health(ctx.tools, path)
        checks.append(Check("sin muestras NaN", nans == 0, f"{nans} NaN"))
        checks.append(Check(f"pico <= {PEAK_LIMIT_DB} dBFS", peak <= PEAK_LIMIT_DB, f"{peak:.2f} dB"))
    return checks


def verify_frames(ctx: Context, entry: dict[str, Any], manifest: dict[str, Any]) -> list[Check]:
    frames = manifest["frames"]
    if not entry["has_video"]:
        return [Check("sin fotogramas", frames == [], str(len(frames)))]
    expected = math.ceil(entry["duration_s"] / ctx.frame_interval_s - 1e-9)
    stamps = [frame["timestamp_s"] for frame in frames]
    checks = [
        Check("número de fotogramas", len(frames) == expected, f"{len(frames)} (esperados {expected})"),
        Check("marcas de tiempo crecientes", all(a < b for a, b in itertools.pairwise(stamps))),
    ]
    if frames:
        first = ctx.client.get(asset_url(manifest["evidence_id"], frames[0]["name"])).content
        checks.append(
            Check("fotograma JPEG íntegro", first.startswith(b"\xff\xd8") and sha256(first) == frames[0]["sha256"])
        )
        if entry["portrait"]:
            checks.append(Check("fotograma vertical", frames[0]["height"] > frames[0]["width"]))
    return checks


def normalize_words(text: str) -> list[str]:
    return [NUMBER_WORDS.get(word, word) for word in re.findall(r"[a-z0-9']+", text.lower())]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (sustituciones + borrados + inserciones) / palabras de la referencia (Levenshtein por palabras)."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1] / max(len(ref), 1)


def verify_transcript(entry: dict[str, Any], manifest: dict[str, Any]) -> TranscriptVerdict:
    mode, transcript = entry["transcript"], manifest["transcript"]
    if mode == "none":
        return TranscriptVerdict((Check("sin transcripción (sin audio)", transcript is None),))
    if transcript is None:
        return TranscriptVerdict((Check("transcripción presente", False),))
    segments = transcript["segments"]
    text = " ".join(segment["text"] for segment in segments)
    if mode == "silence":
        return TranscriptVerdict((Check("silencio sin texto (VAD)", not segments, text[:80]),))
    starts = [segment["start"] for segment in segments]
    checks = [
        Check(
            "segmentos ordenados y dentro de la duración",
            starts == sorted(starts) and all(s["end"] <= entry["duration_s"] + 0.5 for s in segments),
        ),
        Check("modelo CPU int8", (transcript["device"], transcript["compute_type"]) == ("cpu", "int8")),
    ]
    if entry["absent_keywords"]:
        present = [word for word in entry["absent_keywords"] if word in normalize_words(text)]
        checks.append(Check("se transcribe la primera pista", not present, f"aparece {present}"))
    wer = word_error_rate(entry["reference"], text)
    detail = f"WER {wer:.1%}: {text[:140]}"
    if mode == "speech":
        checks.append(Check("idioma en", transcript["language"] == "en", transcript["language"]))
        checks.append(Check(f"WER <= {WER_LIMIT:.0%}", wer <= WER_LIMIT, detail))
        return TranscriptVerdict(tuple(checks), wer=wer)
    soft = (f"{entry['name']}: {detail}",) if wer > WER_LIMIT else ()
    return TranscriptVerdict(tuple(checks), soft, wer)


# --- Ejecución de casos --------------------------------------------------------------------------


def run_case(ctx: Context, entry: dict[str, Any], iteration: int) -> CaseResult:
    content = (ctx.corpus_dir / entry["name"]).read_bytes()
    started = time.perf_counter()
    response = ctx.client.post(
        "/api/evidence",
        params={"filename": entry["name"], "source": "upload"},
        content=content,
        headers={**CSRF, "Content-Type": "application/octet-stream"},
    )
    if entry["outcome"] == "upload_rejected":
        check = Check(
            "rechazado en la subida", response.status_code == entry["expected_status"], f"HTTP {response.status_code}"
        )
        return CaseResult(iteration, entry["name"], time.perf_counter() - started, (check,))
    if response.status_code != 201:
        check = Check("subida aceptada", False, f"HTTP {response.status_code}: {response.text[:200]}")
        return CaseResult(iteration, entry["name"], time.perf_counter() - started, (check,))
    upload = response.json()
    job = wait_for_job(ctx.client, upload["job_id"])
    seconds = time.perf_counter() - started
    checks = [Check("hash calculado en la subida", upload["sha256"] == entry["sha256"])]
    verdict = TranscriptVerdict(())
    if entry["outcome"] == "processing_fails":
        error = job.get("error") or ""
        checks.append(
            Check(
                "falla controlada",
                job["stage"] == "failed" and entry["expected_error"] in error,
                f"{job['stage']}: {error}",
            )
        )
    elif job["stage"] != "done":
        checks.append(Check("procesamiento completado", False, f"{job['stage']}: {job.get('error')}"))
    else:
        manifest = ctx.client.get(f"/api/evidence/{upload['evidence_id']}").json()
        checks.extend(verify_manifest(ctx, entry, manifest))
        verdict = verify_transcript(entry, manifest)
        checks.extend(verdict.checks)
    leftovers = list((ctx.data_dir / "workspace").glob("*"))
    checks.append(Check("workspace vacío tras el trabajo", not leftovers, f"{len(leftovers)} restos"))
    return CaseResult(iteration, entry["name"], seconds, tuple(checks), verdict.soft_misses, verdict.wer)


def verify_manifest(ctx: Context, entry: dict[str, Any], manifest: dict[str, Any]) -> list[Check]:
    media = manifest["media"]
    original = ctx.client.get(asset_url(manifest["evidence_id"], "original.bin")).content
    checks = [
        Check("integridad verificada", manifest["integrity_verified"] is True),
        Check(
            "audio/video detectados",
            (media["has_video"], media["has_audio"]) == (entry["has_video"], entry["has_audio"]),
            f"video={media['has_video']} audio={media['has_audio']}",
        ),
        Check("original intacto", sha256(original) == entry["sha256"]),
        Check("sin campos de riesgo", not any(k in json.dumps(manifest).lower() for k in ("risk", "threat", "score"))),
    ]
    return checks + verify_derivatives(ctx, entry, manifest) + verify_frames(ctx, entry, manifest)


def run_http_negatives(client: httpx.Client, iteration: int) -> CaseResult:
    started = time.perf_counter()
    upload_headers = {**CSRF, "Content-Type": "application/octet-stream"}
    params = {"filename": "a.wav"}
    checks = (
        Check(
            "subida vacía -> 400",
            client.post("/api/evidence", params=params, content=b"", headers=upload_headers).status_code == 400,
        ),
        Check("sin cabecera CSRF -> 403", client.post("/api/evidence", params=params, content=b"x").status_code == 403),
        Check(
            "origen ajeno -> 403",
            client.post(
                "/api/evidence",
                params=params,
                content=b"x",
                headers={**upload_headers, "Origin": "http://evil.example"},
            ).status_code
            == 403,
        ),
        Check(
            "Host no confiable -> 400",
            client.get("/api/health", headers={"Host": "rebinding.example"}).status_code == 400,
        ),
        Check("docs desactivadas -> 404", client.get("/docs").status_code == 404),
    )
    return CaseResult(iteration, "zz_http_negativos", time.perf_counter() - started, checks)


# --- Verificaciones globales y reporte -----------------------------------------------------------------


def global_checks(
    data_dir: Path, log_path: Path, process: subprocess.Popen[bytes], expected_failures: int
) -> list[Check]:
    hits = [
        str(p.relative_to(data_dir))
        for p in data_dir.rglob("*")
        if p.is_file() and any(marker in p.read_bytes() for marker in PLAINTEXT_MARKERS)
    ]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    job_errors = log.count(" ERROR app.jobs: ")
    crash_retries = log.count("de forma anómala")
    return [
        Check("sin texto en claro en la bóveda", not hits, ", ".join(hits[:5])),
        Check("workspace vacío al final", not any((data_dir / "workspace").glob("*"))),
        Check("servidor vivo tras toda la prueba", process.poll() is None),
        Check(
            "errores de trabajo = fallos esperados",
            job_errors == expected_failures,
            f"{job_errors} vs {expected_failures}",
        ),
        Check("caídas nativas de ffmpeg reintentadas (informativo)", True, str(crash_retries)),
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def write_reports(
    results: list[CaseResult], checks: list[Check], meta: dict[str, Any], corpus: list[dict[str, Any]]
) -> Path:
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    payload = {
        "meta": meta,
        "global_checks": [asdict(c) for c in checks],
        "results": [{**asdict(r), "passed": r.passed} for r in results],
    }
    (report_dir / f"e2e_stress_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = report_dir / f"e2e_stress_{stamp}.md"
    markdown.write_text(render_markdown(results, checks, meta, corpus), encoding="utf-8")
    return markdown


def render_markdown(
    results: list[CaseResult], checks: list[Check], meta: dict[str, Any], corpus: list[dict[str, Any]]
) -> str:
    passed = sum(r.passed for r in results)
    total_checks = sum(len(r.checks) for r in results)
    lines = [
        "# Vidence — prueba de estrés de punta a punta",
        "",
        f"- Fecha: {meta['finished_at']} · duración total: {meta['total_minutes']:.1f} min",
        f"- Equipo: {meta['machine']} · Python {meta['python']}",
        f"- Herramientas: {meta['ffmpeg']} · Whisper {meta['whisper']} (CPU int8)",
        f"- Iteraciones: {meta['iterations']} · casos: {len(results)} · verificaciones individuales: {total_checks}",
        "",
        f"## Resultado: {passed}/{len(results)} casos correctos ({100 * passed / len(results):.1f} %)",
        "",
        "| Archivo | Caso | OK | WER medio | Media (s) | p95 (s) | Máx (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    descriptions = {e["name"]: e["description"] for e in corpus} | {
        "zz_http_negativos": "Rechazos HTTP (vacío, CSRF, origen, Host, docs)"
    }
    for name in sorted({r.name for r in results}):
        runs = [r for r in results if r.name == name]
        times = [r.seconds for r in runs]
        wers = [r.wer for r in runs if r.wer is not None]
        wer = f"{statistics.mean(wers):.1%}" if wers else "—"
        lines.append(
            f"| `{name}` | {descriptions.get(name, '')} | {sum(r.passed for r in runs)}/{len(runs)} | {wer} | "
            f"{statistics.mean(times):.1f} | {percentile(times, 0.95):.1f} | {max(times):.1f} |"
        )
    failures = [(r, c) for r in results for c in r.checks if not c.ok]
    lines += ["", f"## Fallos ({len(failures)})", ""]
    lines += [f"- Iteración {r.iteration}, `{r.name}`: **{c.name}** — {c.detail}" for r, c in failures] or ["Ninguno."]
    soft = [miss for r in results for miss in r.soft_misses]
    lines += ["", f"## Avisos no bloqueantes ({len(soft)})", ""]
    lines += [f"- {miss}" for miss in soft] or ["Ninguno."]
    lines += ["", "## Verificaciones globales", ""]
    lines += [f"- {'OK' if c.ok else 'FALLA'} — {c.name}{f': {c.detail}' if c.detail else ''}" for c in checks]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "samples" / "corpus")
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    corpus = json.loads((args.corpus / "corpus.json").read_text(encoding="utf-8"))
    settings = Settings()
    tools = FfmpegTools.resolve(settings.ffmpeg_bin, settings.ffprobe_bin, 600)
    work = Path(tempfile.mkdtemp(prefix="vidence-e2e-"))
    data_dir, scratch, log_path = work / "data", work / "scratch", work / "server.log"
    scratch.mkdir()
    started = time.monotonic()
    results: list[CaseResult] = []
    with (
        httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=300) as client,
        running_server(data_dir, args.port, log_path) as process,
    ):
        health = wait_until_healthy(client, process)
        ctx = Context(client, tools, args.corpus, data_dir, scratch, settings.frame_interval_seconds)
        for iteration in range(1, args.iterations + 1):
            batch = [run_case(ctx, entry, iteration) for entry in corpus] + [run_http_negatives(client, iteration)]
            results.extend(batch)
            for result in (r for r in batch if not r.passed):
                logger.info("  FALLA %s: %s", result.name, [f"{c.name}: {c.detail}" for c in result.checks if not c.ok])
            logger.info(
                "Iteración %d/%d: %d/%d casos OK (%.0f s acumulados)",
                iteration,
                args.iterations,
                sum(r.passed for r in batch),
                len(batch),
                time.monotonic() - started,
            )
        expected_failures = args.iterations * sum(e["outcome"] == "processing_fails" for e in corpus)
        checks = global_checks(data_dir, log_path, process, expected_failures)
    meta = {
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "iterations": args.iterations,
        "total_minutes": (time.monotonic() - started) / 60,
        "machine": f"{platform.processor()} ({os.cpu_count()} hilos)",
        "python": platform.python_version(),
        "ffmpeg": tools.version,
        "whisper": health["whisper_model"],
        "work_dir": str(work),
    }
    report = write_reports(results, checks, meta, corpus)
    ok = all(r.passed for r in results) and all(c.ok for c in checks)
    logger.info("Reporte: %s · resultado: %s", report, "TODO OK" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

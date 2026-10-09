"""Aprovisionamiento único de modelos de IA — ÚNICO paso de Vidence que usa la red.

Descarga los modelos Whisper (formato CTranslate2) y el modelo RNNoise, y
escribe `models/MANIFEST.json` con el SHA-256 de cada archivo. En ejecución,
Vidence trabaja sin red y verifica los modelos contra ese manifiesto.

Este script NO importa el paquete `app` a propósito: `app` instala el guardia
que bloquea la red al importarse.

Uso:
    python scripts/provision_models.py                  # whisper small + tiny, rnnoise
    python scripts/provision_models.py --whisper medium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_NAME = "MANIFEST.json"

RNNOISE_URL = "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/conjoined-burgers-2018-08-28/cb.rnnn"
RNNOISE_FILENAME = "cb.rnnn"
VISION_URL = "https://huggingface.co/Xuban/yolo_weights_database/resolve/main/yolov8n.onnx"
VISION_FILENAME = "yolov8n.onnx"
DOWNLOAD_TIMEOUT_SECONDS = 120
HASH_BLOCK_SIZE = 1024 * 1024

logger = logging.getLogger("provision")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def download_whisper(size: str) -> None:
    from faster_whisper.utils import download_model

    target = MODELS_DIR / "whisper" / size
    logger.info("[whisper] descargando %r -> %s", size, target)
    download_model(size, output_dir=str(target))


def download_rnnoise() -> None:
    target = MODELS_DIR / "rnnoise" / RNNOISE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[rnnoise] descargando %s -> %s", RNNOISE_FILENAME, target)
    with urllib.request.urlopen(RNNOISE_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        payload = response.read()
    if not payload.startswith(b"rnnoise-nu model file"):
        raise RuntimeError("El archivo descargado no es un modelo RNNoise válido")
    target.write_bytes(payload)


def download_vision() -> None:
    target = MODELS_DIR / "vision" / VISION_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[vision] descargando %s -> %s", VISION_FILENAME, target)
    req = urllib.request.Request(VISION_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        payload = response.read()
    if len(payload) < 1_000_000:
        raise RuntimeError("El archivo descargado no es un modelo ONNX válido")
    target.write_bytes(payload)


def write_manifest() -> Path:
    files = {
        path.relative_to(MODELS_DIR).as_posix(): sha256_file(path)
        for path in sorted(MODELS_DIR.rglob("*"))
        if path.is_file()
        and path.name != MANIFEST_NAME
        and not any(part.startswith(".") for part in path.relative_to(MODELS_DIR).parts)
    }
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "files": files}
    manifest_path = MODELS_DIR / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper", nargs="+", default=["small", "tiny"], help="tamaños de Whisper a descargar")
    parser.add_argument("--skip-whisper", action="store_true", help="omitir descarga de Whisper si ya existe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_whisper:
        for size in args.whisper:
            download_whisper(size)
    download_rnnoise()
    download_vision()
    manifest_path = write_manifest()
    logger.info("[ok] manifiesto de integridad escrito en %s", manifest_path)
    logger.info("[ok] Aprovisionamiento terminado. Desde ahora Vidence funciona sin red.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

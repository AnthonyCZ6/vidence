"""Integridad de los modelos de IA: verificación SHA-256 contra models/MANIFEST.json.

El manifiesto lo escribe scripts/provision_models.py en el único paso con red.
Un modelo alterado o ausente del manifiesto no se carga.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"
_HASH_BLOCK_SIZE = 1024 * 1024


class ModelIntegrityError(Exception):
    """Un archivo de modelo no coincide con el manifiesto de integridad."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def verify_model_files(models_dir: Path, prefix: str) -> int:
    """Verifica todos los archivos del manifiesto bajo `prefix`. Devuelve cuántos se verificaron."""
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ModelIntegrityError("Falta models/MANIFEST.json; ejecute scripts/provision_models.py")
    files: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    entries = {name: digest for name, digest in files.items() if name.startswith(prefix)}
    if not entries:
        raise ModelIntegrityError(f"Ningún archivo de '{prefix}' figura en el manifiesto de integridad")
    for relative_name, expected in entries.items():
        path = models_dir / relative_name
        if not path.is_file() or sha256_file(path) != expected:
            raise ModelIntegrityError(f"El archivo de modelo '{relative_name}' falta o fue modificado")
    return len(entries)

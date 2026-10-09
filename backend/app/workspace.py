"""Workspace efímero en claro.

ffmpeg, OpenCV y Whisper necesitan archivos en claro (los MP4/MOV requieren
acceso aleatorio). Este módulo acota su vida: el directorio se borra al
terminar cada trabajo (con o sin error) y al iniciar el servidor.

Limitación: en SSD el borrado lógico no garantiza destrucción física (wear
leveling). La mitigación es el cifrado de disco completo (ver README).
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"^[0-9a-f]{32}$")


@contextmanager
def ephemeral_workspace(root: Path, name: str) -> Iterator[Path]:
    if not _SAFE_NAME_RE.fullmatch(name):
        raise ValueError("Nombre de workspace inválido")
    path = root / name
    _remove(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        _remove(path)


def purge_stale_workspaces(root: Path) -> int:
    """Elimina restos de trabajos interrumpidos (p. ej. tras un corte de energía)."""
    if not root.is_dir():
        return 0
    leftovers = list(root.iterdir())
    for leftover in leftovers:
        _remove(leftover)
    return len(leftovers)


def _remove(path: Path) -> None:
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        logger.exception("No se pudo eliminar el workspace %s", path.name)

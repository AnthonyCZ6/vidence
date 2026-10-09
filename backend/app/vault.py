"""Bóveda local: todo objeto se guarda cifrado (AES-256-GCM por bloques).

Estructura en disco:
    <vault>/vault.json                     parámetros KDF + comprobador de clave (sin secretos)
    <vault>/evidence/<id>/<nombre>.enc     original, derivados, fotogramas y manifiestos

Los nombres en disco nunca provienen del cliente: el id es un UUID generado
aquí y los nombres de objeto se validan con una lista blanca estricta.

Cada objeto se cifra ligado a su ubicación ("<id>/<nombre>" forma parte de la
AAD): copiar un .enc sobre otro, o entre evidencias, hace fallar el descifrado.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

from app.crypto import (
    DEFAULT_CHUNK_SIZE,
    KdfCost,
    KdfParams,
    StreamEncryptor,
    compute_key_check,
    derive_keys,
    inspect_layout,
    iter_decrypt,
    iter_decrypt_range,
    verify_key_check,
)

VAULT_FORMAT = "vidence-vault"
VAULT_VERSION = 1
CONFIG_NAME = "vault.json"
MIN_PASSPHRASE_LENGTH = 10
SALT_SIZE = 16

_EVIDENCE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_OBJECT_NAME_RE = re.compile(r"^[a-z0-9_-]+(/[a-z0-9_-]+)?\.[a-z0-9]{1,5}$")
_SUFFIX = ".enc"
_PARTIAL_SUFFIX = ".partial"

# Límites para parámetros KDF leídos de vault.json (un archivo manipulado no debe
# poder forzar asignaciones de memoria desmedidas).
_KDF_MAX_ITERATIONS = 64
_KDF_MAX_LANES = 64
_KDF_MIN_MEMORY_KIB_PER_LANE = 8
_KDF_MAX_MEMORY_KIB = 4 * 1024 * 1024

logger = logging.getLogger(__name__)


class VaultError(Exception):
    """Error base de la bóveda."""


class InvalidNameError(VaultError):
    """Identificador o nombre de objeto no permitido."""


class ObjectNotFoundError(VaultError):
    """El objeto solicitado no existe en la bóveda."""


class WeakPassphraseError(VaultError):
    """La frase de acceso no cumple el mínimo de seguridad."""


@dataclass(frozen=True)
class StoredObject:
    name: str
    sha256: str
    size_bytes: int


def _object_context(evidence_id: str, name: str) -> bytes:
    return f"vidence/{evidence_id}/{name}".encode("ascii")


def _kdf_params_from_config(kdf: Mapping[str, Any]) -> KdfParams:
    try:
        salt = base64.b64decode(kdf["salt"], validate=True)
        cost = KdfCost(int(kdf["iterations"]), int(kdf["lanes"]), int(kdf["memory_cost_kib"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise VaultError("vault.json tiene parámetros KDF ilegibles") from exc
    in_bounds = (
        len(salt) >= SALT_SIZE
        and 1 <= cost.iterations <= _KDF_MAX_ITERATIONS
        and 1 <= cost.lanes <= _KDF_MAX_LANES
        and _KDF_MIN_MEMORY_KIB_PER_LANE * cost.lanes <= cost.memory_cost_kib <= _KDF_MAX_MEMORY_KIB
    )
    if not in_bounds:
        raise VaultError("vault.json tiene parámetros KDF fuera de rango (¿archivo manipulado?)")
    return KdfParams(salt=salt, cost=cost)


def _new_config(params: KdfParams, key_check: bytes) -> dict[str, Any]:
    return {
        "format": VAULT_FORMAT,
        "version": VAULT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "kdf": {
            "algorithm": "argon2id",
            "salt": base64.b64encode(params.salt).decode("ascii"),
            "iterations": params.cost.iterations,
            "lanes": params.cost.lanes,
            "memory_cost_kib": params.cost.memory_cost_kib,
        },
        "key_check": base64.b64encode(key_check).decode("ascii"),
    }


def _write_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """El salt solo vive en vault.json: un archivo a medio escribir haría perder toda la evidencia."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ObjectWriter:
    """Escritura incremental: calcula el SHA-256 del contenido en claro mientras lo cifra.

    El archivo solo aparece con su nombre final tras `commit()` (reemplazo atómico).
    """

    def __init__(self, name: str, final_path: Path, kek: bytes, chunk_size: int, context: bytes) -> None:
        self._name = name
        self._final_path = final_path
        self._partial_path = final_path.with_name(final_path.name + _PARTIAL_SUFFIX)
        self._handle: BinaryIO = self._partial_path.open("wb")
        self._encryptor = StreamEncryptor(kek, self._handle, chunk_size, context=context)
        self._digest = hashlib.sha256()
        self._size = 0
        self._closed = False
        # La ingesta escribe desde hilos auxiliares; abort() (p. ej. al desconectarse el
        # cliente) debe esperar a la escritura en curso antes de cerrar y borrar.
        self._lock = threading.Lock()

    @property
    def size_bytes(self) -> int:
        return self._size

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._closed:
                raise VaultError("El escritor ya fue cerrado")
            self._digest.update(data)
            self._size += len(data)
            self._encryptor.write(data)

    def commit(self) -> StoredObject:
        with self._lock:
            self._encryptor.finalize()
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True
            os.replace(self._partial_path, self._final_path)
            return StoredObject(self._name, self._digest.hexdigest(), self._size)

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.close()
            self._closed = True
            try:
                self._partial_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("No se pudo eliminar el fragmento cifrado %s", self._partial_path.name)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is not None or not self._closed:
            self.abort()


class Vault:
    def __init__(self, root: Path, kek: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self._root = root
        self._kek = kek
        self._chunk_size = chunk_size

    # --- Apertura ------------------------------------------------------------

    @staticmethod
    def is_initialized(root: Path) -> bool:
        return (root / CONFIG_NAME).is_file()

    @classmethod
    def open_or_create(
        cls,
        root: Path,
        passphrase: str,
        *,
        kdf_cost: KdfCost | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Vault:
        """Desbloquea la bóveda existente o crea una nueva. Lanza InvalidKeyError si la frase no coincide."""
        config_path = root / CONFIG_NAME
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            keys = derive_keys(passphrase, _kdf_params_from_config(config["kdf"]))
            verify_key_check(keys, base64.b64decode(config["key_check"]))
            return cls(root, keys.kek, chunk_size)

        if len(passphrase) < MIN_PASSPHRASE_LENGTH:
            raise WeakPassphraseError(f"La frase de acceso debe tener al menos {MIN_PASSPHRASE_LENGTH} caracteres")
        params = KdfParams(salt=os.urandom(SALT_SIZE), cost=kdf_cost or KdfCost())
        keys = derive_keys(passphrase, params)
        root.mkdir(parents=True, exist_ok=True)
        _write_atomically(config_path, _new_config(params, compute_key_check(keys)))
        return cls(root, keys.kek, chunk_size)

    # --- Rutas seguras ---------------------------------------------------------

    @staticmethod
    def new_evidence_id() -> str:
        return uuid.uuid4().hex

    def _evidence_dir(self, evidence_id: str) -> Path:
        if not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            raise InvalidNameError("Identificador de evidencia inválido")
        return self._root / "evidence" / evidence_id

    def object_path(self, evidence_id: str, name: str) -> Path:
        directory = self._evidence_dir(evidence_id)
        if not _OBJECT_NAME_RE.fullmatch(name):
            raise InvalidNameError("Nombre de objeto inválido")
        return directory / f"{name}{_SUFFIX}"

    def discard_if_empty(self, evidence_id: str) -> None:
        """Elimina la carpeta de una evidencia que quedó sin objetos (p. ej. subida abortada)."""
        directory = self._evidence_dir(evidence_id)
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    def _existing_path(self, evidence_id: str, name: str) -> Path:
        path = self.object_path(evidence_id, name)
        if not path.is_file():
            raise ObjectNotFoundError(f"No existe el objeto {name}")
        return path

    def exists(self, evidence_id: str, name: str) -> bool:
        return self.object_path(evidence_id, name).is_file()

    def list_evidence_ids(self) -> list[str]:
        evidence_root = self._root / "evidence"
        if not evidence_root.is_dir():
            return []
        entries = [p for p in evidence_root.iterdir() if p.is_dir() and _EVIDENCE_ID_RE.fullmatch(p.name)]
        return [p.name for p in sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True)]

    # --- Escritura ---------------------------------------------------------------

    def open_writer(self, evidence_id: str, name: str) -> ObjectWriter:
        path = self.object_path(evidence_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return ObjectWriter(name, path, self._kek, self._chunk_size, _object_context(evidence_id, name))

    def store_bytes(self, evidence_id: str, name: str, data: bytes) -> StoredObject:
        with self.open_writer(evidence_id, name) as writer:
            writer.write(data)
            return writer.commit()

    def store_file(self, evidence_id: str, name: str, source: Path) -> StoredObject:
        with self.open_writer(evidence_id, name) as writer, source.open("rb") as handle:
            while block := handle.read(self._chunk_size):
                writer.write(block)
            return writer.commit()

    def store_json(self, evidence_id: str, name: str, payload: Mapping[str, Any]) -> StoredObject:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        return self.store_bytes(evidence_id, name, data)

    # --- Lectura -----------------------------------------------------------------

    def iter_object(self, evidence_id: str, name: str, byte_range: tuple[int, int] | None = None) -> Iterator[bytes]:
        """Generador de bytes en claro; `byte_range` es (inicio, fin) inclusivo."""
        path = self._existing_path(evidence_id, name)
        context = _object_context(evidence_id, name)
        with path.open("rb") as handle:
            if byte_range is None:
                yield from iter_decrypt(self._kek, handle, context=context)
            else:
                start, end = byte_range
                yield from iter_decrypt_range(self._kek, handle, path.stat().st_size, start, end, context=context)

    def object_size(self, evidence_id: str, name: str) -> int:
        path = self._existing_path(evidence_id, name)
        with path.open("rb") as handle:
            return inspect_layout(handle, path.stat().st_size).plaintext_size

    def read_bytes(self, evidence_id: str, name: str) -> bytes:
        return b"".join(self.iter_object(evidence_id, name))

    def read_json(self, evidence_id: str, name: str) -> Any:
        return json.loads(self.read_bytes(evidence_id, name).decode("utf-8"))

    def decrypt_to_file(self, evidence_id: str, name: str, destination: Path) -> StoredObject:
        """Descifra a un archivo del workspace y devuelve el SHA-256 recalculado."""
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as handle:
            for block in self.iter_object(evidence_id, name):
                digest.update(block)
                size += len(block)
                handle.write(block)
        return StoredObject(name, digest.hexdigest(), size)

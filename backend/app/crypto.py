"""Cifrado autenticado por bloques (AES-256-GCM, construcción STREAM) y derivación de claves.

Formato de archivo `.enc`
-------------------------
Cabecera de 56 bytes (la AAD de cada bloque es cabecera ‖ contexto):

    magic         4  b"VDNC"
    version       1  0x01
    chunk_size    4  uint32 big-endian (tamaño del bloque en claro)
    nonce_prefix  7  aleatorio por archivo
    wrapped_dek  40  DEK de 256 bits envuelta con la KEK (AES Key Wrap, RFC 3394)

Bloques: AES-256-GCM(dek, nonce, bloque, aad=cabecera) = len(bloque) + 16 bytes de tag,
con nonce = nonce_prefix(7) || contador uint32 BE (4) || bandera de último bloque (1).

El contador impide reordenar bloques, la bandera impide truncar en un límite de
bloque y la cabecera como AAD impide alterar parámetros o mezclar archivos. El
contexto (no se almacena: lo aporta quien lee, p. ej. "evidencia/nombre") liga
el archivo a su ubicación, de modo que no puede intercambiarse por otro.
Cada archivo tiene su propia DEK aleatoria: cambiar la frase de acceso solo
requiere re-envolver DEKs, no re-cifrar videos.
"""

from __future__ import annotations

import os
import struct
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.keywrap import InvalidUnwrap, aes_key_unwrap, aes_key_wrap

MAGIC = b"VDNC"
FORMAT_VERSION = 1
KEY_SIZE = 32
TAG_SIZE = 16
NONCE_PREFIX_SIZE = 7
WRAPPED_KEY_SIZE = 40
DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_CHUNK_SIZE = 16 * 1024 * 1024
MAX_CHUNKS = 2**32

_HEADER_STRUCT = struct.Struct(f">4sBI{NONCE_PREFIX_SIZE}s{WRAPPED_KEY_SIZE}s")
HEADER_SIZE = _HEADER_STRUCT.size

_KEY_CHECK_LABEL = b"vidence/key-check/v1"


class CryptoError(Exception):
    """Error base del módulo de cifrado."""


class FormatError(CryptoError):
    """El archivo no tiene el formato cifrado de Vidence."""


class InvalidKeyError(CryptoError):
    """La clave (o frase de acceso) no corresponde."""


class IntegrityError(CryptoError):
    """El contenido cifrado fue alterado, truncado o reordenado."""


# --- Formato -----------------------------------------------------------------


@dataclass(frozen=True)
class EncryptedHeader:
    chunk_size: int
    nonce_prefix: bytes
    wrapped_dek: bytes

    def to_bytes(self) -> bytes:
        return _HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, self.chunk_size, self.nonce_prefix, self.wrapped_dek)

    @classmethod
    def from_bytes(cls, raw: bytes) -> EncryptedHeader:
        if len(raw) != HEADER_SIZE:
            raise FormatError("Cabecera cifrada incompleta")
        magic, version, chunk_size, nonce_prefix, wrapped_dek = _HEADER_STRUCT.unpack(raw)
        if magic != MAGIC:
            raise FormatError("No es un archivo cifrado de Vidence")
        if version != FORMAT_VERSION:
            raise FormatError(f"Versión de formato no soportada: {version}")
        if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
            raise FormatError(f"Tamaño de bloque inválido: {chunk_size}")
        return cls(chunk_size, nonce_prefix, wrapped_dek)


@dataclass(frozen=True)
class EncryptedLayout:
    header: EncryptedHeader
    chunk_count: int
    plaintext_size: int


def generate_key() -> bytes:
    return AESGCM.generate_key(bit_length=KEY_SIZE * 8)


def _require_key(key: bytes) -> None:
    if len(key) != KEY_SIZE:
        raise CryptoError("Se requiere una clave de 256 bits")


def _chunk_nonce(prefix: bytes, index: int, last: bool) -> bytes:
    if index >= MAX_CHUNKS:
        raise CryptoError("Archivo demasiado grande para el formato")
    return prefix + index.to_bytes(4, "big") + (b"\x01" if last else b"\x00")


# --- Cifrado -----------------------------------------------------------------


class StreamEncryptor:
    """Cifra un flujo de longitud desconocida; retiene el último bloque hasta `finalize()`."""

    def __init__(
        self, kek: bytes, sink: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE, *, context: bytes = b""
    ) -> None:
        _require_key(kek)
        if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
            raise CryptoError(f"Tamaño de bloque inválido: {chunk_size}")
        dek = generate_key()
        header = EncryptedHeader(chunk_size, os.urandom(NONCE_PREFIX_SIZE), aes_key_wrap(kek, dek))
        header_bytes = header.to_bytes()
        self._header = header
        self._aad = header_bytes + context
        self._aead = AESGCM(dek)
        self._sink = sink
        self._buffer = bytearray()
        self._index = 0
        self._finalized = False
        sink.write(header_bytes)

    def write(self, data: bytes) -> None:
        if self._finalized:
            raise CryptoError("El cifrador ya fue finalizado")
        self._buffer += data
        chunk_size = self._header.chunk_size
        # Estrictamente mayor: un bloque completo final debe llevar la bandera de último.
        while len(self._buffer) > chunk_size:
            self._emit(bytes(self._buffer[:chunk_size]), last=False)
            del self._buffer[:chunk_size]

    def finalize(self) -> None:
        if self._finalized:
            raise CryptoError("El cifrador ya fue finalizado")
        self._emit(bytes(self._buffer), last=True)
        self._buffer.clear()
        self._finalized = True

    def _emit(self, chunk: bytes, *, last: bool) -> None:
        nonce = _chunk_nonce(self._header.nonce_prefix, self._index, last)
        self._sink.write(self._aead.encrypt(nonce, chunk, self._aad))
        self._index += 1


def encrypt_stream(
    kek: bytes, source: BinaryIO, sink: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE, *, context: bytes = b""
) -> None:
    encryptor = StreamEncryptor(kek, sink, chunk_size, context=context)
    while piece := source.read(chunk_size):
        encryptor.write(piece)
    encryptor.finalize()


# --- Descifrado --------------------------------------------------------------


def read_header(source: BinaryIO) -> EncryptedHeader:
    return EncryptedHeader.from_bytes(source.read(HEADER_SIZE))


def _open_aead(kek: bytes, header: EncryptedHeader) -> AESGCM:
    _require_key(kek)
    try:
        return AESGCM(aes_key_unwrap(kek, header.wrapped_dek))
    except InvalidUnwrap as exc:
        raise InvalidKeyError("La clave no corresponde a este archivo") from exc


def _decrypt_chunk(
    aead: AESGCM, header: EncryptedHeader, aad: bytes, index: int, last: bool, ciphertext: bytes
) -> bytes:
    nonce = _chunk_nonce(header.nonce_prefix, index, last)
    try:
        return aead.decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise IntegrityError(f"Bloque {index} alterado, truncado, fuera de orden o de otra ubicación") from exc


def iter_decrypt(kek: bytes, source: BinaryIO, *, context: bytes = b"") -> Iterator[bytes]:
    """Descifra secuencialmente y verifica la integridad de todo el archivo."""
    header = read_header(source)
    aead = _open_aead(kek, header)
    aad = header.to_bytes() + context
    ciphertext_size = header.chunk_size + TAG_SIZE
    current = source.read(ciphertext_size)
    index = 0
    while True:
        following = source.read(ciphertext_size)
        last = not following
        yield _decrypt_chunk(aead, header, aad, index, last, current)
        if last:
            return
        current, index = following, index + 1


def inspect_layout(source: BinaryIO, file_size: int) -> EncryptedLayout:
    """Calcula número de bloques y tamaño en claro a partir del tamaño cifrado."""
    source.seek(0)
    header = read_header(source)
    body_size = file_size - HEADER_SIZE
    ciphertext_size = header.chunk_size + TAG_SIZE
    if body_size < TAG_SIZE:
        raise IntegrityError("Archivo cifrado truncado")
    chunk_count = -(-body_size // ciphertext_size)
    last_chunk_size = body_size - (chunk_count - 1) * ciphertext_size
    if last_chunk_size < TAG_SIZE:
        raise IntegrityError("Archivo cifrado con un fragmento final inválido")
    return EncryptedLayout(header, chunk_count, body_size - chunk_count * TAG_SIZE)


def iter_decrypt_range(
    kek: bytes, source: BinaryIO, file_size: int, start: int, end: int, *, context: bytes = b""
) -> Iterator[bytes]:
    """Descifra solo los bloques que cubren [start, end] (inclusive). Verifica cada bloque leído."""
    layout = inspect_layout(source, file_size)
    if not 0 <= start <= end < layout.plaintext_size:
        raise ValueError(f"Rango fuera del contenido: {start}-{end}/{layout.plaintext_size}")
    header = layout.header
    aead = _open_aead(kek, header)
    aad = header.to_bytes() + context
    chunk_size = header.chunk_size
    first_index, last_index = start // chunk_size, end // chunk_size
    for index in range(first_index, last_index + 1):
        source.seek(HEADER_SIZE + index * (chunk_size + TAG_SIZE))
        is_last = index == layout.chunk_count - 1
        plaintext = _decrypt_chunk(aead, header, aad, index, is_last, source.read(chunk_size + TAG_SIZE))
        offset = index * chunk_size
        lower = start - offset if index == first_index else 0
        upper = end - offset + 1 if index == last_index else len(plaintext)
        yield plaintext[lower:upper]


# --- Derivación de claves desde la frase de acceso -------------------------------


@dataclass(frozen=True)
class KdfCost:
    """Coste de Argon2id. Por defecto ~RFC 9106 con memoria elevada (256 MiB)."""

    iterations: int = 3
    lanes: int = 4
    memory_cost_kib: int = 256 * 1024


@dataclass(frozen=True)
class KdfParams:
    salt: bytes
    cost: KdfCost = field(default_factory=KdfCost)


@dataclass(frozen=True)
class DerivedKeys:
    kek: bytes = field(repr=False)
    check_key: bytes = field(repr=False)


def derive_keys(passphrase: str, params: KdfParams) -> DerivedKeys:
    """Argon2id → 64 bytes: 32 para la KEK y 32 independientes para verificar la frase.

    La frase se normaliza (NFKC) para que "ñ" compuesta o descompuesta, según el
    teclado o sistema operativo, derive la misma clave.
    """
    passphrase = unicodedata.normalize("NFKC", passphrase)
    if not passphrase:
        raise ValueError("La frase de acceso no puede estar vacía")
    material = Argon2id(
        salt=params.salt,
        length=2 * KEY_SIZE,
        iterations=params.cost.iterations,
        lanes=params.cost.lanes,
        memory_cost=params.cost.memory_cost_kib,
    ).derive(passphrase.encode("utf-8"))
    return DerivedKeys(kek=material[:KEY_SIZE], check_key=material[KEY_SIZE:])


def compute_key_check(keys: DerivedKeys) -> bytes:
    mac = hmac.HMAC(keys.check_key, hashes.SHA256())
    mac.update(_KEY_CHECK_LABEL)
    return mac.finalize()


def verify_key_check(keys: DerivedKeys, expected: bytes) -> None:
    mac = hmac.HMAC(keys.check_key, hashes.SHA256())
    mac.update(_KEY_CHECK_LABEL)
    try:
        mac.verify(expected)
    except InvalidSignature as exc:
        raise InvalidKeyError("Frase de acceso incorrecta") from exc

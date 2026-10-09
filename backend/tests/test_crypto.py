"""Cifrado por bloques AES-256-GCM (construcción STREAM) y derivación de claves."""

from __future__ import annotations

import io
import os

import pytest

from app.crypto import (
    HEADER_SIZE,
    TAG_SIZE,
    CryptoError,
    FormatError,
    IntegrityError,
    InvalidKeyError,
    KdfCost,
    KdfParams,
    StreamEncryptor,
    compute_key_check,
    derive_keys,
    encrypt_stream,
    generate_key,
    inspect_layout,
    iter_decrypt,
    iter_decrypt_range,
    verify_key_check,
)

CHUNK = 64  # bloques diminutos para ejercitar los límites entre bloques
CT_CHUNK = CHUNK + TAG_SIZE
FAST_KDF = KdfCost(iterations=1, lanes=1, memory_cost_kib=8 * 1024)


def _encrypt(kek: bytes, data: bytes) -> bytes:
    sink = io.BytesIO()
    encrypt_stream(kek, io.BytesIO(data), sink, chunk_size=CHUNK)
    return sink.getvalue()


def _decrypt(kek: bytes, blob: bytes) -> bytes:
    return b"".join(iter_decrypt(kek, io.BytesIO(blob)))


@pytest.fixture
def kek() -> bytes:
    return generate_key()


# --- Ida y vuelta ---------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK, 3 * CHUNK + 7])
def test_roundtrip_preserves_content_for_boundary_sizes(kek: bytes, size: int) -> None:
    data = os.urandom(size)
    assert _decrypt(kek, _encrypt(kek, data)) == data


def test_ciphertext_does_not_contain_plaintext(kek: bytes) -> None:
    blob = _encrypt(kek, b"evidencia-secreta " * 20)
    assert b"evidencia-secreta" not in blob


def test_same_plaintext_encrypts_differently_each_time(kek: bytes) -> None:
    data = b"x" * 100
    assert _encrypt(kek, data) != _encrypt(kek, data)


def test_incremental_writer_accepts_arbitrary_piece_sizes(kek: bytes) -> None:
    data = os.urandom(5 * CHUNK + 3)
    sink = io.BytesIO()
    encryptor = StreamEncryptor(kek, sink, chunk_size=CHUNK)
    for piece in (data[:1], data[1 : 2 * CHUNK + 5], data[2 * CHUNK + 5 :]):
        encryptor.write(piece)
    encryptor.finalize()
    assert _decrypt(kek, sink.getvalue()) == data


def test_writer_rejects_writes_after_finalize(kek: bytes) -> None:
    encryptor = StreamEncryptor(kek, io.BytesIO(), chunk_size=CHUNK)
    encryptor.finalize()
    with pytest.raises(CryptoError):
        encryptor.write(b"x")


def test_writer_rejects_keys_that_are_not_256_bits() -> None:
    with pytest.raises(CryptoError):
        StreamEncryptor(b"clave-corta", io.BytesIO())


# --- Detección de manipulación ---------------------------------------------


def test_wrong_key_is_reported_as_invalid_key(kek: bytes) -> None:
    blob = _encrypt(kek, b"datos")
    with pytest.raises(InvalidKeyError):
        _decrypt(generate_key(), blob)


def test_flipped_ciphertext_bit_is_detected(kek: bytes) -> None:
    blob = bytearray(_encrypt(kek, os.urandom(3 * CHUNK)))
    blob[HEADER_SIZE + CT_CHUNK + 5] ^= 0x01  # dentro del segundo bloque
    with pytest.raises(IntegrityError):
        _decrypt(kek, bytes(blob))


def test_modified_header_is_detected(kek: bytes) -> None:
    blob = bytearray(_encrypt(kek, os.urandom(2 * CHUNK)))
    blob[10] ^= 0x01  # byte del prefijo de nonce (la cabecera es AAD)
    with pytest.raises(IntegrityError):
        _decrypt(kek, bytes(blob))


def test_truncation_at_chunk_boundary_is_detected(kek: bytes) -> None:
    blob = _encrypt(kek, os.urandom(3 * CHUNK + 10))
    with pytest.raises(IntegrityError):
        _decrypt(kek, blob[: HEADER_SIZE + 2 * CT_CHUNK])


def test_partial_last_chunk_is_detected(kek: bytes) -> None:
    blob = _encrypt(kek, os.urandom(2 * CHUNK + 10))
    with pytest.raises(IntegrityError):
        _decrypt(kek, blob[:-3])


def test_appended_data_is_detected(kek: bytes) -> None:
    blob = _encrypt(kek, os.urandom(2 * CHUNK))
    with pytest.raises(IntegrityError):
        _decrypt(kek, blob + os.urandom(CT_CHUNK))


def test_reordered_chunks_are_detected(kek: bytes) -> None:
    blob = _encrypt(kek, os.urandom(3 * CHUNK + 1))
    body = blob[HEADER_SIZE:]
    swapped = blob[:HEADER_SIZE] + body[CT_CHUNK : 2 * CT_CHUNK] + body[:CT_CHUNK] + body[2 * CT_CHUNK :]
    with pytest.raises(IntegrityError):
        _decrypt(kek, swapped)


def test_chunks_cannot_be_transplanted_between_files(kek: bytes) -> None:
    first = _encrypt(kek, os.urandom(2 * CHUNK))
    second = _encrypt(kek, os.urandom(2 * CHUNK))
    with pytest.raises(IntegrityError):
        _decrypt(kek, first[:HEADER_SIZE] + second[HEADER_SIZE:])


def test_header_only_file_is_rejected(kek: bytes) -> None:
    blob = _encrypt(kek, b"")
    with pytest.raises(IntegrityError):
        _decrypt(kek, blob[:HEADER_SIZE])


@pytest.mark.parametrize("blob", [b"", b"VDNC", b"NOPE" + bytes(60)])
def test_malformed_header_is_rejected(kek: bytes, blob: bytes) -> None:
    with pytest.raises(FormatError):
        _decrypt(kek, blob)


def test_context_binds_ciphertext_to_its_location(kek: bytes) -> None:
    sink = io.BytesIO()
    encrypt_stream(kek, io.BytesIO(b"contenido"), sink, chunk_size=CHUNK, context=b"evidencia-a/original.bin")
    blob = sink.getvalue()

    assert b"".join(iter_decrypt(kek, io.BytesIO(blob), context=b"evidencia-a/original.bin")) == b"contenido"
    with pytest.raises(IntegrityError):
        b"".join(iter_decrypt(kek, io.BytesIO(blob), context=b"evidencia-b/original.bin"))
    with pytest.raises(IntegrityError):
        list(iter_decrypt_range(kek, io.BytesIO(blob), len(blob), 0, 3, context=b"otro"))


# --- Acceso aleatorio (HTTP Range) -----------------------------------------


@pytest.mark.parametrize("size", [0, 1, CHUNK, 3 * CHUNK + 7])
def test_layout_reports_plaintext_size(kek: bytes, size: int) -> None:
    blob = _encrypt(kek, os.urandom(size))
    layout = inspect_layout(io.BytesIO(blob), len(blob))
    assert layout.plaintext_size == size


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 0), (0, 10), (CHUNK - 1, CHUNK), (5, 3 * CHUNK + 6), (2 * CHUNK, 2 * CHUNK + 3), (3 * CHUNK, 3 * CHUNK + 6)],
)
def test_range_decrypt_returns_exact_slice(kek: bytes, start: int, end: int) -> None:
    data = os.urandom(3 * CHUNK + 7)
    blob = _encrypt(kek, data)
    out = b"".join(iter_decrypt_range(kek, io.BytesIO(blob), len(blob), start, end))
    assert out == data[start : end + 1]


@pytest.mark.parametrize(("start", "end"), [(5, 10), (-1, 3), (6, 5)])
def test_range_outside_content_is_rejected(kek: bytes, start: int, end: int) -> None:
    blob = _encrypt(kek, os.urandom(10))
    with pytest.raises(ValueError):
        list(iter_decrypt_range(kek, io.BytesIO(blob), len(blob), start, end))


def test_range_read_detects_tampering_in_requested_chunk(kek: bytes) -> None:
    blob = bytearray(_encrypt(kek, os.urandom(3 * CHUNK)))
    blob[HEADER_SIZE + CT_CHUNK + 1] ^= 0x01
    with pytest.raises(IntegrityError):
        list(iter_decrypt_range(kek, io.BytesIO(bytes(blob)), len(blob), CHUNK, CHUNK + 5))


def test_layout_rejects_dangling_fragment(kek: bytes) -> None:
    blob = _encrypt(kek, os.urandom(CHUNK)) + b"\x00" * 3
    with pytest.raises(IntegrityError):
        inspect_layout(io.BytesIO(blob), len(blob))


# --- Derivación de claves (Argon2id) ----------------------------------------


def _params() -> KdfParams:
    return KdfParams(salt=b"s" * 16, cost=FAST_KDF)


def test_same_passphrase_and_salt_derive_same_keys() -> None:
    assert derive_keys("frase correcta", _params()) == derive_keys("frase correcta", _params())


def test_encryption_and_check_keys_are_separated() -> None:
    keys = derive_keys("frase correcta", _params())
    assert len(keys.kek) == 32
    assert keys.kek != keys.check_key


def test_key_check_accepts_right_and_rejects_wrong_passphrase() -> None:
    good = derive_keys("frase correcta", _params())
    check = compute_key_check(good)
    verify_key_check(good, check)
    with pytest.raises(InvalidKeyError):
        verify_key_check(derive_keys("frase incorrecta", _params()), check)


def test_unicode_equivalent_passphrases_derive_the_same_key() -> None:
    composed = derive_keys("contraseña segura", _params())
    decomposed = derive_keys("contraseña segura", _params())
    assert composed == decomposed


def test_empty_passphrase_is_rejected() -> None:
    with pytest.raises(ValueError):
        derive_keys("", _params())

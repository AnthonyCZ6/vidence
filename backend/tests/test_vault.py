"""Bóveda cifrada: almacenamiento, hashes, nombres seguros y frase de acceso."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from app.crypto import IntegrityError, InvalidKeyError
from app.vault import InvalidNameError, ObjectNotFoundError, Vault, VaultError, WeakPassphraseError
from tests.conftest import FAST_KDF, TEST_PASSPHRASE


def _open(root: Path, passphrase: str = TEST_PASSPHRASE) -> Vault:
    return Vault.open_or_create(root, passphrase, kdf_cost=FAST_KDF, chunk_size=1024)


# --- Frase de acceso ----------------------------------------------------------


def test_reopening_with_same_passphrase_reads_existing_objects(tmp_path: Path) -> None:
    first = _open(tmp_path / "vault")
    evidence_id = first.new_evidence_id()
    first.store_bytes(evidence_id, "nota.txt", b"hola")

    second = _open(tmp_path / "vault")
    assert second.read_bytes(evidence_id, "nota.txt") == b"hola"


def test_wrong_passphrase_is_rejected(tmp_path: Path) -> None:
    _open(tmp_path / "vault")
    with pytest.raises(InvalidKeyError):
        _open(tmp_path / "vault", "otra-frase-distinta")


def test_short_passphrase_is_rejected_on_creation(tmp_path: Path) -> None:
    with pytest.raises(WeakPassphraseError):
        _open(tmp_path / "vault", "corta")


def test_vault_config_contains_no_key_material(tmp_path: Path) -> None:
    _open(tmp_path / "vault")
    config_text = (tmp_path / "vault" / "vault.json").read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert set(config) == {"format", "version", "created_at", "kdf", "key_check"}
    assert TEST_PASSPHRASE not in config_text
    assert Vault.is_initialized(tmp_path / "vault")


# --- Objetos cifrados ---------------------------------------------------------


def test_store_bytes_returns_plaintext_hash_and_size(vault: Vault) -> None:
    stored = vault.store_bytes(vault.new_evidence_id(), "a.bin", b"abc")
    assert stored.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert stored.size_bytes == 3


def test_objects_are_encrypted_on_disk(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    vault.store_bytes(evidence_id, "a.txt", b"texto en claro " * 50)
    raw = vault.object_path(evidence_id, "a.txt").read_bytes()
    assert b"texto en claro" not in raw


def test_writer_is_invisible_until_committed(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    writer = vault.open_writer(evidence_id, "original.bin")
    writer.write(b"parte-1 ")
    writer.write(b"parte-2")
    assert not vault.exists(evidence_id, "original.bin")
    stored = writer.commit()
    assert vault.exists(evidence_id, "original.bin")
    assert vault.read_bytes(evidence_id, "original.bin") == b"parte-1 parte-2"
    assert stored.size_bytes == len(b"parte-1 parte-2")


def test_aborted_writer_leaves_no_files(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    writer = vault.open_writer(evidence_id, "original.bin")
    writer.write(b"x" * 5000)
    writer.abort()
    evidence_dir = vault.object_path(evidence_id, "original.bin").parent
    assert not any(evidence_dir.iterdir())


def test_writer_context_manager_aborts_on_exception(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    with pytest.raises(RuntimeError), vault.open_writer(evidence_id, "original.bin") as writer:
        writer.write(b"x")
        raise RuntimeError("fallo a mitad de la subida")
    assert not vault.exists(evidence_id, "original.bin")


def test_file_roundtrip_through_vault(vault: Vault, tmp_path: Path) -> None:
    data = os.urandom(10_000)
    source = tmp_path / "entrada.bin"
    source.write_bytes(data)
    evidence_id = vault.new_evidence_id()

    stored = vault.store_file(evidence_id, "original.bin", source)
    restored = vault.decrypt_to_file(evidence_id, "original.bin", tmp_path / "salida.bin")

    assert (tmp_path / "salida.bin").read_bytes() == data
    assert restored.sha256 == stored.sha256 == hashlib.sha256(data).hexdigest()


def test_json_roundtrip_keeps_unicode(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    payload = {"archivo": "grabación ñandú.m4a", "duracion": 12.5}
    vault.store_json(evidence_id, "manifest.json", payload)
    assert vault.read_json(evidence_id, "manifest.json") == payload


def test_iter_object_supports_byte_ranges(vault: Vault) -> None:
    data = os.urandom(5000)
    evidence_id = vault.new_evidence_id()
    vault.store_bytes(evidence_id, "video.mp4", data)

    assert vault.object_size(evidence_id, "video.mp4") == 5000
    assert b"".join(vault.iter_object(evidence_id, "video.mp4")) == data
    assert b"".join(vault.iter_object(evidence_id, "video.mp4", (100, 4999))) == data[100:]


def test_nested_frame_names_are_supported(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    vault.store_bytes(evidence_id, "frames/0001.jpg", b"\xff\xd8jpeg")
    assert vault.read_bytes(evidence_id, "frames/0001.jpg") == b"\xff\xd8jpeg"


def test_objects_cannot_be_swapped_within_an_evidence(vault: Vault) -> None:
    evidence_id = vault.new_evidence_id()
    vault.store_bytes(evidence_id, "enhanced.flac", b"mejorado")
    vault.store_bytes(evidence_id, "original_preview.flac", b"vista previa")
    target = vault.object_path(evidence_id, "original_preview.flac")
    target.write_bytes(vault.object_path(evidence_id, "enhanced.flac").read_bytes())
    with pytest.raises(IntegrityError):
        vault.read_bytes(evidence_id, "original_preview.flac")


def test_objects_cannot_be_moved_between_evidences(vault: Vault) -> None:
    first, second = vault.new_evidence_id(), vault.new_evidence_id()
    vault.store_json(first, "manifest.json", {"evidence_id": first})
    vault.store_json(second, "manifest.json", {"evidence_id": second})
    vault.object_path(second, "manifest.json").write_bytes(vault.object_path(first, "manifest.json").read_bytes())
    with pytest.raises(IntegrityError):
        vault.read_json(second, "manifest.json")


def test_tampered_kdf_parameters_are_rejected(tmp_path: Path) -> None:
    _open(tmp_path / "vault")
    config_path = tmp_path / "vault" / "vault.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["kdf"]["memory_cost_kib"] = 2**40
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(VaultError):
        _open(tmp_path / "vault")


# --- Nombres seguros (sin path traversal) --------------------------------------


@pytest.mark.parametrize(
    "bad_name", ["../escape.txt", "a/b/c.txt", "MAYUS.txt", "sin_extension", "frames/../x.jpg", "x.extlarga", ""]
)
def test_invalid_object_names_are_rejected(vault: Vault, bad_name: str) -> None:
    with pytest.raises(InvalidNameError):
        vault.store_bytes(vault.new_evidence_id(), bad_name, b"x")


@pytest.mark.parametrize("bad_id", ["..", "no-es-un-id", "0" * 31, "G" * 32, "../" + "0" * 29])
def test_invalid_evidence_ids_are_rejected(vault: Vault, bad_id: str) -> None:
    with pytest.raises(InvalidNameError):
        vault.read_bytes(bad_id, "a.txt")


def test_missing_object_raises_not_found(vault: Vault) -> None:
    with pytest.raises(ObjectNotFoundError):
        vault.read_bytes(vault.new_evidence_id(), "inexistente.txt")


def test_list_evidence_ids_returns_newest_first(vault: Vault) -> None:
    older, newer = vault.new_evidence_id(), vault.new_evidence_id()
    vault.store_bytes(older, "a.txt", b"1")
    vault.store_bytes(newer, "a.txt", b"2")
    old_dir = vault.object_path(older, "a.txt").parent
    os.utime(old_dir, (1_000_000, 1_000_000))
    assert vault.list_evidence_ids()[:2] == [newer, older]

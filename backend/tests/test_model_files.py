"""Verificación de integridad de los modelos de IA contra MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.model_files import ModelIntegrityError, verify_model_files


def _models(tmp_path: Path, content: bytes = b"pesos") -> Path:
    root = tmp_path / "models"
    model_file = root / "whisper" / "tiny" / "model.bin"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(content)
    manifest = {"files": {"whisper/tiny/model.bin": hashlib.sha256(b"pesos").hexdigest()}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_untouched_files_pass_verification(tmp_path: Path) -> None:
    assert verify_model_files(_models(tmp_path), "whisper/tiny/") == 1


def test_modified_model_file_is_detected(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError):
        verify_model_files(_models(tmp_path, b"pesos-manipulados"), "whisper/tiny/")


def test_model_absent_from_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError):
        verify_model_files(_models(tmp_path), "whisper/small/")


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    root = _models(tmp_path)
    (root / "MANIFEST.json").unlink()
    with pytest.raises(ModelIntegrityError):
        verify_model_files(root, "whisper/tiny/")

"""Workspace efímero: los datos en claro se eliminan siempre."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.workspace import ephemeral_workspace, purge_stale_workspaces

NAME = "0123456789abcdef0123456789abcdef"


def test_workspace_is_removed_after_success(tmp_path: Path) -> None:
    with ephemeral_workspace(tmp_path, NAME) as workspace:
        (workspace / "original.mp4").write_bytes(b"en claro")
        assert workspace.is_dir()
    assert not (tmp_path / NAME).exists()


def test_workspace_is_removed_after_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), ephemeral_workspace(tmp_path, NAME) as workspace:
        (workspace / "original.mp4").write_bytes(b"en claro")
        raise RuntimeError("fallo del pipeline")
    assert not (tmp_path / NAME).exists()


def test_stale_workspaces_are_purged(tmp_path: Path) -> None:
    leftover = tmp_path / NAME
    leftover.mkdir()
    (leftover / "restos.wav").write_bytes(b"en claro")
    assert purge_stale_workspaces(tmp_path) == 1
    assert not any(tmp_path.iterdir())


def test_purge_tolerates_missing_root(tmp_path: Path) -> None:
    assert purge_stale_workspaces(tmp_path / "no-existe") == 0


@pytest.mark.parametrize("bad_name", ["../fuera", "", "ABC", "x" * 32])
def test_unsafe_workspace_names_are_rejected(tmp_path: Path, bad_name: str) -> None:
    with pytest.raises(ValueError), ephemeral_workspace(tmp_path, bad_name):
        pass

"""Configuración (variables VIDENCE_* o backend/.env). Inmutable una vez cargada."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

# Nombres simples: sin separadores de ruta ni "..", aptos para filtros de ffmpeg.
SIMPLE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDENCE_",
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    frontend_dir: Path = PROJECT_ROOT / "frontend"

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    trusted_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_timeout_seconds: int = Field(default=3600, ge=10)

    whisper_model: str = Field(default="small", pattern=SIMPLE_NAME_PATTERN)
    whisper_language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$")
    whisper_cpu_threads: int = Field(default=0, ge=0)
    whisper_beam_size: int = Field(default=5, ge=1, le=10)

    rnnoise_model: str = Field(default="cb.rnnn", pattern=SIMPLE_NAME_PATTERN)
    vision_model: str = Field(default="yolov8n.onnx", pattern=SIMPLE_NAME_PATTERN)
    vision_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    vision_nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    frame_interval_seconds: float = Field(default=5.0, gt=0)
    max_frames: int = Field(default=240, ge=1, le=2000)
    max_upload_bytes: int = Field(default=4 * 1024**3, ge=1)

    @property
    def vault_dir(self) -> Path:
        return self.data_dir / "vault"

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @property
    def rnnoise_dir(self) -> Path:
        return self.models_dir / "rnnoise"

    @property
    def rnnoise_path(self) -> Path:
        return self.rnnoise_dir / self.rnnoise_model

    @property
    def vision_dir(self) -> Path:
        return self.models_dir / "vision"

    @property
    def vision_path(self) -> Path:
        return self.vision_dir / self.vision_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

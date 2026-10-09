"""Punto de entrada: `python -m app`.

Pide la frase de acceso de la bóveda (o la crea en el primer arranque),
deriva la clave con Argon2id y sirve la aplicación solo en loopback.
La frase nunca se escribe en disco; puede pasarse por la variable de entorno
VIDENCE_PASSPHRASE (se elimina del entorno al leerla).
"""

from __future__ import annotations

import getpass
import ipaddress
import logging
import os
import sys

import uvicorn

from app.config import Settings, get_settings
from app.crypto import InvalidKeyError
from app.main import create_app
from app.media import MediaError
from app.model_files import ModelIntegrityError
from app.vault import MIN_PASSPHRASE_LENGTH, Vault, VaultError

logger = logging.getLogger("vidence")

PASSPHRASE_ENV = "VIDENCE_PASSPHRASE"  # noqa: S105  (nombre de la variable, no un secreto)


class PassphraseMismatchError(ValueError):
    """Las dos frases introducidas no coinciden."""


def _read_passphrase(settings: Settings) -> str:
    from_env = os.environ.pop(PASSPHRASE_ENV, None)
    if from_env:
        return from_env
    if Vault.is_initialized(settings.vault_dir):
        return getpass.getpass("Frase de acceso de la bóveda: ")
    logger.info("Primer arranque: se creará una bóveda cifrada nueva en %s", settings.vault_dir)
    logger.info("Si pierde la frase de acceso, la evidencia no podrá recuperarse.")
    first = getpass.getpass(f"Nueva frase de acceso (mínimo {MIN_PASSPHRASE_LENGTH} caracteres): ")
    if getpass.getpass("Repita la frase de acceso: ") != first:
        raise PassphraseMismatchError("Las frases no coinciden")
    return first


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    try:
        vault = Vault.open_or_create(settings.vault_dir, _read_passphrase(settings))
        app = create_app(settings, vault)
    except InvalidKeyError:
        logger.error("Frase de acceso incorrecta.")
        return 1
    except (PassphraseMismatchError, VaultError, MediaError, ModelIntegrityError) as exc:
        logger.error("%s", exc)
        return 1

    if not _is_loopback(settings.host):
        logger.warning("Escuchando en %s: el MVP no tiene autenticación; úselo solo en loopback.", settings.host)
    logger.info("Vidence listo en http://%s:%d", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False, server_header=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

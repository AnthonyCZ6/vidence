"""Air-gap por software: bloquea cualquier salida de red del proceso Python.

Sustituye las operaciones de red del módulo `socket` (connect, connect_ex,
sendto, sendmsg, getaddrinfo, getnameinfo, gethostbyname*, gethostbyaddr) y,
en Windows, `connect`/`sendto` del proactor IOCP de asyncio (que llama a
ConnectEx/WSASendTo sin pasar por `socket`) por versiones que solo permiten
loopback. Todas las librerías Python (huggingface_hub, httpx, urllib,
asyncio) pasan por ellas. Se instala al importar `app`, antes que cualquier
dependencia.

Límites: el código nativo que abre sockets por su cuenta (ffmpeg, extensiones
en C/Rust) no pasa por `socket`. Para ffmpeg se usa `-protocol_whitelist
file`, y para todo el proceso, la regla de firewall saliente del README.

Se descartó un audit hook (PEP 578): en CPython 3.14, con un hook activo,
`marshal.loads` se corrompe en algunos procesos (`python -m`, `-c`) y la
carga de .pyc falla, lo que impedía arrancar la aplicación.
"""

from __future__ import annotations

import functools
import ipaddress
import os
import socket
import sys
from collections.abc import Callable
from typing import Any

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OTEL_SDK_DISABLED": "true",
    "DO_NOT_TRACK": "1",
}

_LOCAL_NAMES = frozenset({"", "localhost"})
_SENDMSG_ADDRESS_INDEX = 3

_installed = False

HostExtractor = Callable[[tuple[Any, ...], dict[str, Any]], object]


class EgressBlockedError(PermissionError):
    """Se intentó una comunicación de red fuera del equipo."""


def is_local_host(host: object) -> bool:
    """True si `host` es loopback (o vacío/None, que significa "este equipo")."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if not isinstance(host, str):
        return False
    name = host.strip().strip("[]").split("%", 1)[0].lower()
    if name in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def check_destination(host: object, operation: str) -> None:
    if not is_local_host(host):
        raise EgressBlockedError(f"Vidence offline: comunicación de red bloqueada ({operation} -> {host!r})")


def _host_from_address(address: object) -> object:
    # AF_INET/AF_INET6 usan tuplas (host, puerto, ...); AF_UNIX usa una ruta local.
    if isinstance(address, tuple) and address:
        return address[0]
    if isinstance(address, str | bytes):
        return None
    return address


def _hostname(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    return args[0] if args else kwargs.get("host")


def _first_address(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    return _host_from_address(args[0] if args else kwargs.get("address"))


def _last_address(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> object:
    return _host_from_address(args[-1]) if args else None


def _sendmsg_address(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> object:
    # Sin dirección explícita, el socket ya pasó por connect() (y fue verificado allí).
    return _host_from_address(args[_SENDMSG_ADDRESS_INDEX]) if len(args) > _SENDMSG_ADDRESS_INDEX else None


def _guarded(original: Callable[..., Any], operation: str, extract_host: HostExtractor, *, is_method: bool) -> Any:
    @functools.wraps(original)
    def guard(*args: Any, **kwargs: Any) -> Any:
        check_destination(extract_host(args[1:] if is_method else args, kwargs), operation)
        return original(*args, **kwargs)

    return guard


def _guard_iocp_proactor() -> None:
    """En Windows, asyncio conecta y envía UDP con IOCP (ConnectEx/WSASendTo), no con `socket`."""
    if sys.platform != "win32":
        return
    from asyncio import windows_events

    proactor = windows_events.IocpProactor
    connect, sendto = proactor.connect, proactor.sendto

    @functools.wraps(connect)
    def guarded_connect(self: Any, conn: socket.socket, address: Any) -> Any:
        check_destination(_host_from_address(address), "asyncio.IocpProactor.connect")
        return connect(self, conn, address)

    @functools.wraps(sendto)
    def guarded_sendto(self: Any, conn: socket.socket, buf: bytes, flags: int = 0, addr: Any = None) -> Any:
        check_destination(_host_from_address(addr), "asyncio.IocpProactor.sendto")
        return sendto(self, conn, buf, flags, addr)

    proactor.connect = guarded_connect  # type: ignore[method-assign]
    proactor.sendto = guarded_sendto  # type: ignore[method-assign]


def install() -> None:
    """Fuerza el modo offline de las librerías y bloquea la red no local (idempotente)."""
    global _installed
    if _installed:
        return
    os.environ.update(OFFLINE_ENV)
    methods = (
        ("connect", _first_address),
        ("connect_ex", _first_address),
        ("sendto", _last_address),
        ("sendmsg", _sendmsg_address),
    )
    for name, extract_host in methods:
        if hasattr(socket.socket, name):  # sendmsg no existe en Windows
            setattr(
                socket.socket,
                name,
                _guarded(getattr(socket.socket, name), f"socket.{name}", extract_host, is_method=True),
            )
    functions = (
        ("getaddrinfo", _hostname),
        ("gethostbyname", _hostname),
        ("gethostbyname_ex", _hostname),
        ("gethostbyaddr", _hostname),
        ("getnameinfo", _first_address),
    )
    for name, extract_host in functions:
        setattr(socket, name, _guarded(getattr(socket, name), f"socket.{name}", extract_host, is_method=False))
    _guard_iocp_proactor()
    _installed = True


def is_installed() -> bool:
    return _installed

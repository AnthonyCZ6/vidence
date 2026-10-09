"""Guardia offline: ninguna conexión puede salir del equipo."""

from __future__ import annotations

import asyncio
import os
import socket
import sys

import pytest

from app import offline_guard
from app.offline_guard import OFFLINE_ENV, EgressBlockedError, check_destination, is_local_host

UNROUTABLE_TEST_NET = "203.0.113.7"  # RFC 5737 TEST-NET-3


@pytest.mark.parametrize(
    "host", [None, "", "localhost", "LOCALHOST", "127.0.0.1", "127.8.9.10", "::1", "[::1]", b"127.0.0.1"]
)
def test_loopback_hosts_are_local(host: object) -> None:
    assert is_local_host(host)


@pytest.mark.parametrize(
    "host", [UNROUTABLE_TEST_NET, "8.8.8.8", "huggingface.co", "example.com", "192.168.1.10", "10.0.0.1", 12345]
)
def test_remote_hosts_are_not_local(host: object) -> None:
    assert not is_local_host(host)


def test_check_destination_blocks_remote_and_allows_loopback() -> None:
    check_destination("127.0.0.1", "prueba")
    with pytest.raises(EgressBlockedError):
        check_destination("huggingface.co", "prueba")


def test_guard_is_installed_when_package_is_imported() -> None:
    assert offline_guard.is_installed()


def test_offline_environment_is_forced() -> None:
    for key, value in OFFLINE_ENV.items():
        assert os.environ[key] == value


# --- Sockets reales: el bloqueo ocurre antes de enviar un solo paquete ----------------


def test_tcp_connect_to_internet_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(EgressBlockedError):
        sock.connect((UNROUTABLE_TEST_NET, 9))


def test_tcp_connect_ex_to_internet_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(EgressBlockedError):
        sock.connect_ex((UNROUTABLE_TEST_NET, 9))


def test_udp_sendto_internet_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, pytest.raises(EgressBlockedError):
        sock.sendto(b"consulta", ("8.8.8.8", 53))


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        (socket.getaddrinfo, ("huggingface.co", 443)),
        (socket.gethostbyname, ("example.com",)),
        (socket.gethostbyname_ex, ("example.com",)),
        (socket.gethostbyaddr, ("8.8.8.8",)),
        (socket.getnameinfo, (("8.8.8.8", 53), 0)),
        (socket.create_connection, ((UNROUTABLE_TEST_NET, 443),)),
    ],
)
def test_name_resolution_and_helpers_are_blocked(function: object, argument: tuple[object, ...]) -> None:
    with pytest.raises(EgressBlockedError):
        function(*argument)  # type: ignore[operator]


def test_asyncio_connection_to_ip_literal_is_blocked() -> None:
    async def connect() -> None:
        await asyncio.open_connection(UNROUTABLE_TEST_NET, 443)

    with pytest.raises(EgressBlockedError):
        asyncio.run(connect())


@pytest.mark.skipif(sys.platform != "win32", reason="IOCP solo existe en Windows")
def test_asyncio_udp_sendto_internet_is_blocked() -> None:
    async def send() -> None:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, local_addr=("0.0.0.0", 0))
        try:
            await loop._proactor.sendto(transport.get_extra_info("socket")._sock, b"consulta", addr=("8.8.8.8", 53))  # type: ignore[attr-defined]
        finally:
            transport.close()

    with pytest.raises(EgressBlockedError):
        asyncio.run(send())


def test_loopback_connections_still_work() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass


def test_asyncio_loopback_connections_still_work() -> None:
    async def roundtrip() -> bytes:
        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(await reader.read(4))
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"ping")
            await writer.drain()
            data = await reader.read(4)
            writer.close()
            return data

    assert asyncio.run(roundtrip()) == b"ping"

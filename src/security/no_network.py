"""Blocco esplicito chiamate di rete outbound (local-first)."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Callable


class NetworkBlockedError(RuntimeError):
    """Errore lanciato quando una chiamata outbound è bloccata."""


@dataclass
class NetworkGuardState:
    """Stato patch socket per ripristino."""

    original_socket_connect: Callable
    original_create_connection: Callable


_GUARD_STATE: NetworkGuardState | None = None


def _is_loopback(host: str) -> bool:
    host = host.strip().lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def block_outbound_network(allow_loopback: bool = True) -> None:
    """Blocca connessioni outbound via socket.connect/create_connection."""

    global _GUARD_STATE
    if _GUARD_STATE is not None:
        return

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) and address else ""
        if allow_loopback and isinstance(host, str) and _is_loopback(host):
            return original_connect(self, address)
        raise NetworkBlockedError(f"Outbound network blocked: connect({address})")

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) and address else ""
        if allow_loopback and isinstance(host, str) and _is_loopback(host):
            return original_create_connection(address, *args, **kwargs)
        raise NetworkBlockedError(f"Outbound network blocked: create_connection({address})")

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create_connection
    _GUARD_STATE = NetworkGuardState(
        original_socket_connect=original_connect,
        original_create_connection=original_create_connection,
    )


def restore_network() -> None:
    """Ripristina comportamento socket originale."""

    global _GUARD_STATE
    if _GUARD_STATE is None:
        return

    socket.socket.connect = _GUARD_STATE.original_socket_connect
    socket.create_connection = _GUARD_STATE.original_create_connection
    _GUARD_STATE = None

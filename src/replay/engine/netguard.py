"""Defense in depth during replay: block outbound connections to hosts that aren't allowed.

Tools are mocked, so a replay should only ever talk to LLM providers (and the judge). If an
uninstrumented code path tries to reach a payments or email API, this stops it. It works at the
level of Python sockets, so it cannot see C-extension clients, subprocesses or file writes.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from replay.errors import ReplayNetworkBlocked


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _host_allowed(host: str, allowed: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in (h.lower() for h in allowed))


@contextmanager
def network_guard(allowed_hosts: Callable[[], Iterable[str]]) -> Iterator[None]:
    """Only let sockets reach `allowed_hosts()` (and their subdomains) while active.

    `allowed_hosts` is called on each lookup, so clients wrapped mid-replay are allowed too.
    """
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    resolved: set[str] = set()
    lock = threading.Lock()

    def blocked(target: str) -> ReplayNetworkBlocked:
        return ReplayNetworkBlocked(
            f"Replay blocked a network connection to {target}: during a replay only LLM hosts "
            "and network.allow_hosts may be reached. If this is a read-only service, add it to "
            "network.allow_hosts; if it's a tool, decorate it with @replay.tool."
        )

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode() if isinstance(host, bytes) else host
        if isinstance(name, str) and not _is_ip(name):
            if not _host_allowed(name, allowed_hosts()):
                raise blocked(name)
            infos = real_getaddrinfo(host, *args, **kwargs)
            with lock:
                resolved.update(str(info[4][0]) for info in infos)
            return infos
        return real_getaddrinfo(host, *args, **kwargs)

    def check(sock: socket.socket, address: Any) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            ip = str(address[0])
            with lock:
                known = ip in resolved
            if not known and not _host_allowed(ip, allowed_hosts()):
                raise blocked(ip)

    def connect(self: socket.socket, address: Any) -> None:
        check(self, address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        check(self, address)
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]

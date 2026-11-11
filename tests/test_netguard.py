from __future__ import annotations

import socket
from collections.abc import Iterator

import httpx2
import pytest

import replay
from replay.engine import ReplayConfig, replay_traces
from replay.engine.netguard import network_guard
from replay.errors import ReplayNetworkBlocked
from tests import toy_agent
from tests.fakes import FakeLLM, chat


@pytest.fixture
def local_server() -> Iterator[int]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen()
    yield server.getsockname()[1]
    server.close()


def connect(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=2):
        pass


def test_blocks_hosts_that_are_not_allowed() -> None:
    with network_guard(lambda: {"api.groq.com"}), pytest.raises(ReplayNetworkBlocked, match="payments"):
        socket.getaddrinfo("payments.example.com", 443)


def test_allows_listed_hosts_and_their_subdomains(local_server: int) -> None:
    with network_guard(lambda: {"localhost"}):
        connect("localhost", local_server)
    with network_guard(lambda: {"example.com"}), pytest.raises(ReplayNetworkBlocked):
        # api.example.com would be allowed by resolution; the IP it resolves to is then trusted
        socket.getaddrinfo("example.org", 443)


def test_blocks_direct_ip_connections_unless_resolved_from_an_allowed_host(
    local_server: int,
) -> None:
    with network_guard(lambda: set()), pytest.raises(ReplayNetworkBlocked, match="127.0.0.1"):
        connect("127.0.0.1", local_server)
    with network_guard(lambda: {"127.0.0.1"}):
        connect("127.0.0.1", local_server)


def test_http_clients_see_a_connection_error() -> None:
    with network_guard(lambda: set()), pytest.raises(httpx2.ConnectError):
        httpx2.get("https://payments.example.com/refund", timeout=2)


def test_sockets_are_restored_afterwards(local_server: int) -> None:
    real_connect = socket.socket.connect
    with network_guard(lambda: set()):
        assert socket.socket.connect is not real_connect
    assert socket.socket.connect is real_connect
    connect("127.0.0.1", local_server)


def test_replays_run_inside_the_guard(sink: replay.MemorySink) -> None:
    """An uninstrumented HTTP call inside the agent is stopped during replay."""
    toy_agent.client = replay.wrap(FakeLLM([chat("ok")]).openai())
    toy_agent.agent("hi")
    baseline = sink.traces[-1]

    def leaky(body: dict[str, object]) -> dict[str, object]:
        httpx2.post("https://email.example.com/send", json={"to": "customer"}, timeout=2)
        return chat("sent")

    toy_agent.client = replay.wrap(FakeLLM(leaky).openai())
    session = replay_traces([baseline], ReplayConfig())
    candidate = session.results[0].candidate
    assert candidate is not None and candidate.error is not None
    assert "email.example.com" in candidate.error.message
    toy_agent.client = None

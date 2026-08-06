from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


@dataclass
class _FakeRemoteMethod:
    answers: list[dict[str, HostAndPort]]
    requested_names: list[str] = field(default_factory=list)

    def remote(self, worker_name: str) -> Any:
        self.requested_names.append(worker_name)
        return _resolved(self.answers[len(self.requested_names) - 1])


@dataclass
class _FakeManagerHandle:
    get_worker_addrs: _FakeRemoteMethod


async def _resolved(value: dict[str, HostAndPort]) -> dict[str, HostAndPort]:
    return value


def _make_handle(*answers: dict[str, HostAndPort]) -> _FakeManagerHandle:
    return _FakeManagerHandle(get_worker_addrs=_FakeRemoteMethod(answers=list(answers)))


class TestRayWorkerProviderAddressLookup:
    async def test_every_lookup_asks_the_manager_again(self):
        """Addresses are never cached, so a relaunched worker is not answered with a stale endpoint."""
        handle = _make_handle(
            {"primary": HostAndPort(host="10.0.0.7", port=15000)},
            {"primary": HostAndPort(host="10.0.0.7", port=15001)},
        )
        provider = RayWorkerProvider(worker_manager_handle=handle, spec_names=["inference-engine-0-0"])

        first = (await provider.get_addrs(worker_name="router-0-0"))["primary"]
        second = (await provider.get_addrs(worker_name="router-0-0"))["primary"]

        assert (first.port, second.port) == (15000, 15001)
        assert handle.get_worker_addrs.requested_names == ["router-0-0", "router-0-0"]


class TestRayWorkerProviderGetAddrs:
    async def test_returns_every_named_port_of_the_worker(self):
        """Consumers that need more than the primary endpoint get the worker's whole address map."""
        addrs = {
            "primary": HostAndPort(host="10.0.0.7", port=15000),
            "disaggregation_bootstrap": HostAndPort(host="10.0.0.7", port=15001),
        }
        handle = _make_handle(addrs)
        provider = RayWorkerProvider(worker_manager_handle=handle, spec_names=["inference-engine-0-0"])

        assert await provider.get_addrs(worker_name="engine-0-0") == addrs


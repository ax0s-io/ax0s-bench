from __future__ import annotations

import ipaddress
import sys
from typing import Any


class EgressGuard:
    """Fail-closed audit hook proving a condition only talks to this host.

    Installed before any memory backend is constructed. Every ``socket.connect``
    to a non-loopback IP address raises, so a silent OpenAI/telemetry fallback
    aborts the run instead of producing a quietly non-local receipt.
    """

    def __init__(self) -> None:
        self.allowed: dict[str, int] = {}
        self.blocked: list[str] = []
        self._installed = False

    @staticmethod
    def _is_loopback(host: str) -> bool:
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host in {"localhost", ""}

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True

        def hook(event: str, args: tuple[Any, ...]) -> None:
            if event != "socket.connect" or len(args) < 2:
                return
            address = args[1]
            if not isinstance(address, tuple) or not address:
                return  # AF_UNIX and friends stay local by construction
            host = str(address[0])
            key = f"{host}:{address[1] if len(address) > 1 else ''}"
            if self._is_loopback(host):
                self.allowed[key] = self.allowed.get(key, 0) + 1
                return
            self.blocked.append(key)
            raise RuntimeError(f"egress guard: blocked non-loopback connection to {key}")

        sys.addaudithook(hook)

    def disclosure(self) -> dict[str, Any]:
        return {
            "policy": "sys.addaudithook on socket.connect; loopback allowed, every other address raises",
            "installed": self._installed,
            "loopback_endpoints": dict(sorted(self.allowed.items())),
            "blocked_endpoints": sorted(set(self.blocked)),
        }

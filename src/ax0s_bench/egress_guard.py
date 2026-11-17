from __future__ import annotations

import ipaddress
import sys
from typing import Any


class EgressGuard:
    """Optional guard for audited Python socket connects.

    When installed, the hook raises on audited ``socket.connect`` events with
    non-loopback tuple addresses. It can catch Python network fallbacks, but
    does not constrain subprocesses or unaudited network activity. It is not
    proof of host-only communication or an OS network policy.
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

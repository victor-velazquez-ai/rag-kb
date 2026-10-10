"""Is a request from this machine? Host and Origin checks shared by the API guard and the MCP guard.

Both guards trust the Host header, which a browser cannot forge for another site. A tunnel or proxy
that rewrites Host to localhost makes remote requests look local, so put real sign-in in front of any
exposed instance.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def host_is_loopback(host: str | None) -> bool:
    """True for a Host header (with or without a port) that names this machine: localhost, 127.0.0.0/8, ::1."""
    if not host:
        return False
    value = host.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return False
        name = value[1:end]
    elif value.count(":") == 1:
        name = value.split(":", 1)[0]
    else:
        name = value
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def origin_is_loopback(origin: str | None) -> bool:
    """An absent Origin is fine (not a browser); a present one must point at this machine."""
    if origin is None:
        return True
    try:
        netloc = urlsplit(origin).netloc
    except ValueError:
        return False
    return host_is_loopback(netloc)

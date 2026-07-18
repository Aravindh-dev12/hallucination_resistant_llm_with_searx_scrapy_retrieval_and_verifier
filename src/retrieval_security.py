"""Security checks for untrusted retrieval inputs."""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

INJECTION_PATTERNS = (
    re.compile(r"ignore (?:all |any )?(?:previous|prior) instructions", re.I),
    re.compile(r"(?:reveal|print|return).{0,30}(?:system prompt|secret|api key)", re.I),
    re.compile(r"(?:you are now|act as).{0,40}(?:administrator|system)", re.I),
)


def detect_prompt_injection(text: str) -> list[str]:
    return [pattern.pattern for pattern in INJECTION_PATTERNS if pattern.search(text)]


def sanitize_evidence(text: str, *, max_characters: int = 20_000) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_characters]


def validate_public_url(url: str, *, resolve_dns: bool = False) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only absolute HTTP(S) URLs are allowed")
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("local network targets are forbidden")
    addresses = []
    try:
        addresses.append(ipaddress.ip_address(hostname))
    except ValueError:
        if resolve_dns:
            addresses.extend(
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(hostname, parsed.port or 443)
            )
    for address in addresses:
        if not address.is_global:
            raise ValueError("non-public network targets are forbidden")
    return url

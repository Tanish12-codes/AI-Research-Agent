"""Security module for prompt-injection defense and content sanitization."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urljoin, urlparse


@dataclass
class SecurityResult:
    """Result of security analysis on untrusted content."""

    is_safe: bool
    cleaned_text: str
    flags: list[str] = field(default_factory=list)
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    trust_reduction: float = 0.0
    removed_sentences: list[str] = field(default_factory=list)


INJECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "instruction_override": [
        re.compile(r"(?i)ignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)"),
        re.compile(r"(?i)disregard\s+(all\s+)?previous\s+(instructions?|context)"),
        re.compile(r"(?i)forget\s+(everything|all)\s+(above|before|previous)"),
        re.compile(r"(?i)override\s+(system|previous|all)\s+(prompt|instructions?)"),
        re.compile(r"(?i)new\s+instructions?\s*:"),
    ],
    "role_manipulation": [
        re.compile(r"(?i)you\s+are\s+now\s+(a|an|the)\s+"),
        re.compile(r"(?i)act\s+as\s+(a|an|the|if)\s+"),
        re.compile(r"(?i)pretend\s+(you|to\s+be)\s+"),
        re.compile(r"(?i)switch\s+to\s+\w+\s+mode"),
        re.compile(r"(?i)enter\s+(developer|debug|admin|sudo|root)\s+mode"),
    ],
    "system_prompt_extraction": [
        re.compile(r"(?i)repeat\s+(your|the)\s+(system\s+)?prompt"),
        re.compile(r"(?i)show\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)"),
        re.compile(r"(?i)what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)"),
        re.compile(r"(?i)reveal\s+(your|the)\s+(hidden|secret|system)"),
    ],
    "hidden_instructions": [
        re.compile(r"(?i)\[\s*(?:system|admin|assistant)\s*(?:message|note|instruction)\s*\]"),
        re.compile(r"(?i)<\s*(?:system|admin|assistant)\s*>"),
        re.compile(r"(?i)###\s*(?:SYSTEM|ADMIN|INTERNAL)\s*###"),
        re.compile(r"(?i)BEGIN\s+(?:SYSTEM|HIDDEN|SECRET)\s+(?:PROMPT|MESSAGE|INSTRUCTION)"),
        re.compile(r"(?i)(?:IMPORTANT|NOTE)\s+(?:FOR|TO)\s+(?:THE\s+)?(?:AI|ASSISTANT|MODEL|LLM|GPT|CLAUDE)"),
    ],
    "data_exfiltration": [
        re.compile(r"(?i)!\[\w*\]\(https?://[^)]*(?:api[_-]?key|token|secret|password|auth)[^)]*\)"),
        re.compile(r"(?i)fetch\s+(?:this\s+)?url"),
        re.compile(r"(?i)send\s+(?:a\s+)?(?:request|data)\s+to"),
        re.compile(r"(?i)load\s+(?:image|resource)\s+from"),
        re.compile(r"!\[(?:[^\]]*)\]\(https?://(?:[^)]*(?:\?|&)[^)]*=)[^)]*\)"),
    ],
    "encoding_evasion": [
        re.compile(r"(?i)base64[\s_-]?(?:decode|encode)"),
        re.compile(r"(?i)rot13"),
        re.compile(r"(?i)decode\s+(?:the\s+)?(?:following|this|below)"),
        re.compile(r"(?i)hex[\s_-]?(?:decode|encode)"),
    ],
}

DANGEROUS_UNICODE: set[str] = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2060",
    "\u2061",
    "\u2062",
    "\u2063",
    "\u2064",
    "\ufeff",
    "\ufffe",
    "\uffff",
}

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|auth[_-]?key|access[_-]?key)\s*[=:]\s*\S+"),
]

APPROVED_PORTS: set[int] = {80, 443, 8080, 8443}


def sanitize_html(raw_html: str) -> str:
    """Strip scripts, styles, hidden elements, HTML comments, and irrelevant markup."""
    if not raw_html:
        return ""

    try:
        from bs4 import BeautifulSoup, Comment

        try:
            soup = BeautifulSoup(raw_html, "lxml")
        except Exception:
            soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup.find_all(["script", "style", "noscript", "iframe", "object", "embed", "applet"]):
            tag.decompose()

        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)):
            tag.decompose()
        for tag in soup.find_all(attrs={"aria-hidden": "true"}):
            tag.decompose()
        for tag in soup.find_all(attrs={"hidden": True}):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

    except ImportError:
        cleaned = re.sub(
            r"(?is)<(script|style|noscript|iframe|object|embed|applet)[^>]*>.*?</\1>",
            "",
            raw_html,
        )
        cleaned = re.sub(r"(?s)<!--.*?-->", "", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        text = html.unescape(cleaned)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize Unicode and remove zero-width/bidirectional control characters."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    text = "".join(c for c in text if c not in DANGEROUS_UNICODE)

    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("C") or c in "\n\r\t "
    )

    return text


def detect_injection_patterns(text: str) -> list[tuple[str, str]]:
    """Detect injection patterns in text. Returns list of (family, matched_context)."""
    detections: list[tuple[str, str]] = []
    if not text:
        return detections

    for family, patterns in INJECTION_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                detections.append((family, context))

    return detections


def quarantine_decision(
    detections: list[tuple[str, str]], threshold: int = 2
) -> tuple[bool, Optional[str]]:
    """Decide whether to quarantine based on number of independent attack indicators."""
    families_triggered = {family for family, _ in detections}

    if len(families_triggered) >= threshold:
        reason = f"Multiple injection pattern families detected: {', '.join(sorted(families_triggered))}"
        return True, reason

    return False, None


def remove_flagged_sentences(
    text: str, detections: list[tuple[str, str]]
) -> tuple[str, list[str]]:
    """Remove sentences containing injection patterns. Returns (cleaned_text, removed_sentences)."""
    if not text or not detections:
        return text, []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    removed: list[str] = []
    kept: list[str] = []

    for sentence in sentences:
        should_remove = False
        for family, patterns in INJECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(sentence):
                    should_remove = True
                    break
            if should_remove:
                break

        if should_remove:
            removed.append(sentence)
        else:
            kept.append(sentence)

    return " ".join(kept), removed


def calculate_trust_reduction(detections: list[tuple[str, str]]) -> float:
    """Calculate trust reduction based on detection severity."""
    if not detections:
        return 0.0

    families = {family for family, _ in detections}

    reduction_per_family: dict[str, float] = {
        "instruction_override": 0.3,
        "role_manipulation": 0.25,
        "system_prompt_extraction": 0.2,
        "hidden_instructions": 0.3,
        "data_exfiltration": 0.4,
        "encoding_evasion": 0.15,
    }

    total = sum(reduction_per_family.get(f, 0.1) for f in families)
    return min(total, 0.9)


def analyze_content_security(
    raw_text: str, is_html: bool = False, defenses_enabled: bool = True
) -> SecurityResult:
    """Full security analysis pipeline for untrusted content."""
    if not raw_text:
        return SecurityResult(
            is_safe=True,
            cleaned_text="",
            flags=[],
            quarantined=False,
            trust_reduction=0.0,
        )

    if not defenses_enabled:
        return SecurityResult(
            is_safe=True,
            cleaned_text=raw_text,
            flags=[],
            quarantined=False,
            trust_reduction=0.0,
        )

    if is_html:
        text = sanitize_html(raw_text)
    else:
        text = raw_text

    text = normalize_unicode(text)

    detections = detect_injection_patterns(text)
    flags = [f"{family}: {context}" for family, context in detections]

    should_quarantine, quarantine_reason = quarantine_decision(detections)

    if should_quarantine:
        return SecurityResult(
            is_safe=False,
            cleaned_text=text,
            flags=flags,
            quarantined=True,
            quarantine_reason=quarantine_reason,
            trust_reduction=0.9,
        )

    cleaned_text, removed = remove_flagged_sentences(text, detections)
    trust_reduction = calculate_trust_reduction(detections)

    return SecurityResult(
        is_safe=len(detections) == 0,
        cleaned_text=cleaned_text,
        flags=flags,
        quarantined=False,
        trust_reduction=trust_reduction,
        removed_sentences=removed,
    )


EVIDENCE_DELIMITER_PATTERN = re.compile(r"(?i)=+\s*(BEGIN|END)\s+EVIDENCE")


def wrap_evidence_block(text: str, evidence_id: str) -> str:
    """Wrap untrusted evidence in clearly delimited blocks."""
    defanged = EVIDENCE_DELIMITER_PATTERN.sub(r"[\1 EVIDENCE]", text)
    return (
        f"=== BEGIN EVIDENCE [{evidence_id}] ===\n"
        f"{defanged}\n"
        f"=== END EVIDENCE [{evidence_id}] ==="
    )


def validate_url(url: str) -> tuple[bool, Optional[str]]:
    """Validate URL for safety (SSRF protection)."""
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False, "Failed to parse URL"

    scheme = parsed.scheme.lower() if parsed.scheme else ""
    if scheme not in ("http", "https"):
        return False, f"Scheme '{parsed.scheme}' not allowed; only http/https"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"

    port = parsed.port
    if port and port not in APPROVED_PORTS:
        return False, f"Port {port} not in approved list: {APPROVED_PORTS}"

    ip = _parse_ip_literal(hostname)
    if ip is not None:
        if not _is_public_ip(ip):
            return False, f"IP address {ip} is private/reserved/loopback"
        return True, None

    hostname_lower = hostname.lower().rstrip(".")
    if hostname_lower == "localhost" or hostname_lower.endswith(".localhost"):
        return False, "Localhost access not allowed"
    if hostname_lower.endswith((".local", ".internal")):
        return False, "Internal hostname not allowed"

    return True, None


def _parse_ip_literal(hostname: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Parse standard and non-standard IP notations (e.g. 2130706433, 0x7f000001, 127.1)."""
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(hostname))
    except (OSError, ValueError):
        return None


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def resolve_public_host(hostname: str) -> tuple[bool, Optional[str]]:
    """Resolve a hostname and require every resolved address to be public."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(hostname, None)
    except (OSError, UnicodeError) as e:
        return False, f"DNS resolution failed for {hostname}: {e}"
    if not infos:
        return False, f"No addresses for {hostname}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not _is_public_ip(ip):
            return False, f"{hostname} resolves to non-public address {ip}"
    return True, None


def validate_redirect(original_url: str, redirect_url: str) -> tuple[bool, Optional[str]]:
    """Validate that a redirect destination is safe against SSRF and protocol downgrades."""
    joined_url = urljoin(original_url, redirect_url)

    is_valid, reason = validate_url(joined_url)
    if not is_valid:
        return False, f"Redirect destination unsafe: {reason}"

    orig_parsed = urlparse(original_url)
    redir_parsed = urlparse(joined_url)
    if orig_parsed.scheme.lower() == "https" and redir_parsed.scheme.lower() == "http":
        return False, "HTTPS to HTTP downgrade not allowed"

    return True, None


def sanitize_output_urls(text: str, allowed_domains: Optional[set[str]] = None) -> str:
    """Remove unapproved URLs and markdown image links from generated output."""
    if not text:
        return ""

    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[image removed]", text)

    if allowed_domains:
        normalized_allowed = {d.lower().removeprefix("www.") for d in allowed_domains if d}

        def check_url(match: re.Match[str]) -> str:
            url = match.group(0)
            try:
                parsed = urlparse(url)
                if parsed.hostname:
                    host = parsed.hostname.lower().removeprefix("www.")
                    if host in normalized_allowed or any(
                        host.endswith("." + d) for d in normalized_allowed
                    ):
                        return url
            except Exception:
                pass
            return "[url removed]"

        text = re.sub(r"https?://[^\s)\]]+", check_url, text)

    return text


def redact_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Redact fields containing sensitive credentials or tokens from a dict (for logging)."""
    if not isinstance(data, dict):
        return data

    sensitive_keys = {"key", "token", "secret", "password", "auth", "credential", "api_key"}
    redacted: dict[str, Any] = {}

    for k, v in data.items():
        key_lower = str(k).lower()
        if any(s in key_lower for s in sensitive_keys):
            redacted[k] = "[REDACTED]"
        elif isinstance(v, dict):
            redacted[k] = redact_secrets(v)
        elif isinstance(v, list):
            redacted[k] = [
                redact_secrets(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            redacted[k] = v

    return redacted


def redact_secrets_in_text(text: str) -> str:
    """Redact secret patterns in plain text strings."""
    if not text:
        return ""
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(r"\1=[REDACTED]", redacted)
    return redacted


__all__ = [
    "SecurityResult",
    "INJECTION_PATTERNS",
    "DANGEROUS_UNICODE",
    "SECRET_PATTERNS",
    "APPROVED_PORTS",
    "sanitize_html",
    "normalize_unicode",
    "detect_injection_patterns",
    "quarantine_decision",
    "remove_flagged_sentences",
    "calculate_trust_reduction",
    "analyze_content_security",
    "wrap_evidence_block",
    "validate_url",
    "validate_redirect",
    "resolve_public_host",
    "sanitize_output_urls",
    "redact_secrets",
    "redact_secrets_in_text",
]

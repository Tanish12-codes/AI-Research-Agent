"""Page fetcher with SSRF protection and content extraction."""
import hashlib
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from ..schemas import Evidence
from ..security import resolve_public_host, sanitize_html, validate_redirect, validate_url
from ..cache import FileCache

MAX_RESPONSE_SIZE = 2 * 1024 * 1024
MAX_REDIRECTS = 3

HostResolver = Callable[[str], Awaitable[tuple[bool, Optional[str]]]]


class ResponseTooLarge(Exception):
    pass


async def _read_capped(response: httpx.Response, max_size: int) -> bytes:
    """Read the body, aborting once it exceeds max_size."""
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_size:
        raise ResponseTooLarge(declared)
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_size:
            raise ResponseTooLarge(len(body))
    return bytes(body)


async def fetch_page(
    url: str,
    timeout: int = 15,
    cache: Optional[FileCache] = None,
    max_size: int = MAX_RESPONSE_SIZE,
    resolver: HostResolver = resolve_public_host,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Optional[Evidence]:
    """Fetch a page and extract its text, or return None if it is unsafe or unusable."""
    is_valid, _ = validate_url(url)
    if not is_valid:
        return None

    cache_key = f"fetch:{hashlib.sha256(url.encode()).hexdigest()}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            payload = cached.get("value", cached) if (isinstance(cached, dict) and "value" in cached and "evidence_id" not in cached) else cached
            return Evidence(**payload)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        ) as client:
            current_url = url
            redirect_count = 0
            while True:
                host_ok, _ = await resolver(urlparse(current_url).hostname or "")
                if not host_ok:
                    return None

                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        if not location or redirect_count >= MAX_REDIRECTS:
                            return None
                        next_url = urljoin(current_url, location)
                        is_valid, _ = validate_redirect(current_url, next_url)
                        if not is_valid:
                            return None
                        current_url = next_url
                        redirect_count += 1
                        continue

                    if response.status_code != 200:
                        return None
                    content_type = response.headers.get("content-type", "")
                    if "text" not in content_type and "html" not in content_type:
                        return None

                    body = await _read_capped(response, max_size)
                    raw_content = body.decode(response.encoding or "utf-8", errors="replace")
                    break

        text = sanitize_html(raw_content)[:5000]
        if not text.strip():
            return None

        parsed = urlparse(current_url)
        try:
            soup = BeautifulSoup(raw_content[:10000], "lxml")
        except Exception:
            soup = BeautifulSoup(raw_content[:10000], "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else parsed.netloc

        evidence = Evidence(
            evidence_id=f"fetch-{hashlib.sha256(current_url.encode()).hexdigest()[:12]}",
            url=current_url,
            domain=parsed.netloc,
            title=title[:200],
            text=text,
            retrieval_timestamp=datetime.now(timezone.utc),
            provider="fetch",
            retrieval_metadata={"original_url": url, "redirects": redirect_count},
        )

        if cache:
            cache.set(cache_key, evidence.model_dump(mode="json"))

        return evidence

    except Exception:
        return None

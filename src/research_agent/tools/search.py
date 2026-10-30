"""Web search providers with fallback support."""
import asyncio
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Any
from urllib.parse import urlparse
import httpx

from ..schemas import Evidence
from ..config import Settings
from ..cache import FileCache


class SearchProvider(ABC):
    """Abstract search provider interface."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Execute a search query and return evidence items."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the search provider."""
        ...


class TavilySearchProvider(SearchProvider):
    """Tavily search API provider."""

    def __init__(self, api_key: str, timeout: int = 15, cache: Optional[FileCache] = None):
        self.api_key = api_key
        self.timeout = timeout
        self.cache = cache

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Execute search query using Tavily API with caching support."""
        cache_key = f"tavily:{hashlib.sha256(query.encode()).hexdigest()}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                items: Any = cached if isinstance(cached, list) else (cached.get("items") or cached.get("value"))
                if isinstance(items, list):
                    return [Evidence(**e) for e in items]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_raw_content": False,
                    "include_answer": False,
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
            data = response.json()

        results: list[Evidence] = []
        for i, r in enumerate(data.get("results", [])):
            parsed = urlparse(r.get("url", ""))
            evidence = Evidence(
                evidence_id=f"tavily-{hashlib.sha256(r.get('url', str(i)).encode()).hexdigest()[:12]}",
                url=r.get("url", ""),
                domain=parsed.netloc if parsed.netloc else "unknown",
                title=r.get("title", ""),
                text=r.get("content", "")[:3000],
                retrieval_timestamp=datetime.now(timezone.utc),
                publication_date=None,
                provider="tavily",
                retrieval_metadata={"score": r.get("score"), "query": query},
                relevance_score=r.get("score"),
            )
            results.append(evidence)

        if self.cache and results:
            dumped = [e.model_dump(mode="json") for e in results]
            self.cache.set(cache_key, {"items": dumped, "value": dumped})

        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider (fallback)."""

    def __init__(self, api_key: str, timeout: int = 15, cache: Optional[FileCache] = None):
        self.api_key = api_key
        self.timeout = timeout
        self.cache = cache

    @property
    def name(self) -> str:
        return "brave"

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Execute search query using Brave Search API with caching support."""
        cache_key = f"brave:{hashlib.sha256(query.encode()).hexdigest()}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                items: Any = cached if isinstance(cached, list) else (cached.get("items") or cached.get("value"))
                if isinstance(items, list):
                    return [Evidence(**e) for e in items]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        results: list[Evidence] = []
        for i, r in enumerate(data.get("web", {}).get("results", [])):
            parsed = urlparse(r.get("url", ""))
            evidence = Evidence(
                evidence_id=f"brave-{hashlib.sha256(r.get('url', str(i)).encode()).hexdigest()[:12]}",
                url=r.get("url", ""),
                domain=parsed.netloc if parsed.netloc else "unknown",
                title=r.get("title", ""),
                text=r.get("description", "")[:3000],
                retrieval_timestamp=datetime.now(timezone.utc),
                provider="brave",
                retrieval_metadata={"query": query},
            )
            results.append(evidence)

        if self.cache and results:
            dumped = [e.model_dump(mode="json") for e in results]
            self.cache.set(cache_key, {"items": dumped, "value": dumped})

        return results


class MockSearchProvider(SearchProvider):
    """Mock search provider for offline testing."""

    def __init__(self, responses: Optional[dict[str, list[Evidence]]] = None):
        self._responses = responses or {}
        self._calls: list[str] = []

    @property
    def name(self) -> str:
        return "mock"

    def add_response(self, query_substring: str, results: list[Evidence]) -> None:
        """Add mock results matching a query substring."""
        self._responses[query_substring.lower()] = results

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Return predefined responses or empty list based on query substring matching."""
        self._calls.append(query)
        query_lower = query.lower()

        for key, results in self._responses.items():
            if key in query_lower:
                return results[:max_results]

        return []


_FREE_USER_AGENT = "ResearchAgent/1.0 (educational research agent; https://github.com/)"


def _evidence_id(provider: str, url: str) -> str:
    return f"{provider}-{hashlib.sha256(url.encode()).hexdigest()[:12]}"


def _cached_items(cache: Optional[FileCache], key: str) -> Optional[list[Evidence]]:
    if not cache:
        return None
    cached = cache.get(key)
    if not cached:
        return None
    items: Any = cached if isinstance(cached, list) else (cached.get("items") or cached.get("value"))
    return [Evidence(**e) for e in items] if isinstance(items, list) else None


def _store_items(cache: Optional[FileCache], key: str, results: list[Evidence]) -> None:
    if cache and results:
        dumped = [e.model_dump(mode="json") for e in results]
        cache.set(key, {"items": dumped, "value": dumped})


class DuckDuckGoSearchProvider(SearchProvider):
    """Keyless web search via DuckDuckGo's HTML endpoint.

    No API key or quota, but it is a scraped endpoint: DuckDuckGo may answer with a
    bot challenge instead of results. That case returns no results (not an error)
    so the chain moves on to the next provider.
    """

    ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: int = 15, cache: Optional[FileCache] = None):
        self.timeout = timeout
        self.cache = cache

    @property
    def name(self) -> str:
        return "duckduckgo"

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        cache_key = f"ddg:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = _cached_items(self.cache, cache_key)
        if cached is not None:
            return cached[:max_results]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
            response = await client.post(self.ENDPOINT, data={"q": query, "kl": "us-en"})
            response.raise_for_status()

        results = self.parse_html(response.text, query, max_results)
        _store_items(self.cache, cache_key, results)
        return results

    @staticmethod
    def parse_html(html: str, query: str, max_results: int = 5) -> list[Evidence]:
        """Parse DuckDuckGo HTML results; exposed for offline tests."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        results: list[Evidence] = []
        for node in soup.select("div.result"):
            if "result--ad" in (node.get("class") or []):
                continue
            link = node.select_one("a.result__a")
            snippet = node.select_one(".result__snippet")
            if not link or not snippet:
                continue
            url = _unwrap_ddg_redirect(link.get("href", ""))
            if not url.startswith(("http://", "https://")):
                continue
            text = snippet.get_text(" ", strip=True)
            if not text:
                continue
            results.append(
                Evidence(
                    evidence_id=_evidence_id("ddg", url),
                    url=url,
                    domain=urlparse(url).netloc or "unknown",
                    title=link.get_text(" ", strip=True),
                    text=text[:3000],
                    retrieval_timestamp=datetime.now(timezone.utc),
                    provider="duckduckgo",
                    retrieval_metadata={"query": query, "rank": len(results) + 1},
                )
            )
            if len(results) >= max_results:
                break
        return results


def _unwrap_ddg_redirect(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded target>."""
    from urllib.parse import parse_qs, unquote

    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


class WikipediaSearchProvider(SearchProvider):
    """Keyless encyclopedic search via the MediaWiki API (returns article intros)."""

    ENDPOINT = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout: int = 15, cache: Optional[FileCache] = None):
        self.timeout = timeout
        self.cache = cache

    @property
    def name(self) -> str:
        return "wikipedia"

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        cache_key = f"wikipedia:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = _cached_items(self.cache, cache_key)
        if cached is not None:
            return cached[:max_results]

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(max_results),
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": "max",
            "inprop": "url",
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": _FREE_USER_AGENT}) as client:
            response = await client.get(self.ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()

        results = self.parse_response(data, query)
        _store_items(self.cache, cache_key, results)
        return results[:max_results]

    @staticmethod
    def parse_response(data: dict, query: str) -> list[Evidence]:
        """Parse a MediaWiki generator=search response; exposed for offline tests."""
        pages = (data.get("query") or {}).get("pages") or []
        if isinstance(pages, dict):
            pages = list(pages.values())
        pages = sorted(pages, key=lambda p: p.get("index", 0))
        results: list[Evidence] = []
        for page in pages:
            text = (page.get("extract") or "").strip()
            url = page.get("fullurl") or ""
            if not text or not url:
                continue
            results.append(
                Evidence(
                    evidence_id=_evidence_id("wikipedia", url),
                    url=url,
                    domain=urlparse(url).netloc or "en.wikipedia.org",
                    title=page.get("title", ""),
                    text=text[:3000],
                    retrieval_timestamp=datetime.now(timezone.utc),
                    provider="wikipedia",
                    retrieval_metadata={"query": query, "rank": page.get("index")},
                )
            )
        return results


class SearchWithFallback:
    """Search orchestrator that walks a provider chain.

    A provider that raises, or returns nothing, hands the query to the next one.
    ``primary`` and ``fallback`` are kept as attributes for the common two-provider case.
    """

    def __init__(
        self,
        primary: SearchProvider,
        fallback: Optional[SearchProvider] = None,
        extra_fallbacks: Optional[list[SearchProvider]] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.providers: list[SearchProvider] = [primary]
        if fallback:
            self.providers.append(fallback)
        self.providers.extend(extra_fallbacks or [])
        self.provider_used: str = ""

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Return the first non-empty result set; re-raise only if every provider failed."""
        last_exc: Optional[Exception] = None
        any_succeeded = False
        for provider in self.providers:
            try:
                results = await provider.search(query, max_results)
            except Exception as exc:
                last_exc = exc
                continue
            any_succeeded = True
            self.provider_used = provider.name
            if results:
                return results
        if last_exc is not None and not any_succeeded:
            raise last_exc
        return []


def build_search(settings: Settings, cache: Optional[FileCache] = None) -> Optional[SearchWithFallback]:
    """Chain keyed providers first (better ranking), then keyless free ones.

    Returns None when nothing is available (no keys and free search disabled).
    """
    timeout = settings.search_timeout_seconds
    chain: list[SearchProvider] = []
    if settings.tavily_api_key:
        chain.append(TavilySearchProvider(api_key=settings.tavily_api_key, timeout=timeout, cache=cache))
    if settings.brave_api_key:
        chain.append(BraveSearchProvider(api_key=settings.brave_api_key, timeout=timeout, cache=cache))
    if getattr(settings, "enable_free_search", True):
        chain.append(DuckDuckGoSearchProvider(timeout=timeout, cache=cache))
        chain.append(WikipediaSearchProvider(timeout=timeout, cache=cache))
    if not chain:
        return None
    return SearchWithFallback(chain[0], chain[1] if len(chain) > 1 else None, chain[2:])

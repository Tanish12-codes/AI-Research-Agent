"""Academic paper search: arXiv and Semantic Scholar (both free, no key needed).

Paper abstracts are clean, quotable text from high-authority sources, which suits
the verbatim-quote verifier. The tool is routed deterministically: it runs only for
questions that look academic (see ``is_academic_question``), the same way the
GitHub tool runs only on repository URLs found in search results.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import httpx

from ..cache import FileCache
from ..schemas import Evidence

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,citationCount,url,authors,externalIds,venue,publicationDate"
_USER_AGENT = "ResearchAgent/1.0 (educational research agent)"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}

_ACADEMIC_TERMS = re.compile(
    r"\b(paper|papers|research|study|studies|survey|literature|arxiv|state[- ]of[- ]the[- ]art|sota|"
    r"benchmark\w*|approach(?:es)?|algorithm\w*|method\w*|technique\w*|architecture\w*|"
    r"llm\w*|neural|transformer\w*|rag|retrieval|embedding\w*|agent\w*|machine learning|deep learning|"
    r"reinforcement|quantization|fine[- ]tun\w*|memory)\b",
    re.IGNORECASE,
)


def is_academic_question(question: str) -> bool:
    """Deterministic routing rule: does this question benefit from paper search?"""
    return bool(_ACADEMIC_TERMS.search(question))


_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "with", "vs", "versus",
    "what", "how", "which", "are", "is", "best", "top", "using", "use", "between", "compare",
    "comparison", "pros", "cons", "different", "approaches", "research", "adding",
}


def arxiv_query(query: str, max_terms: int = 4) -> str:
    """AND the most specific terms: plain `all:a b c` matches papers containing ANY term."""
    terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", query) if t.lower() not in _STOP and len(t) > 2]
    if not terms:
        return f"all:{query}"
    return " AND ".join(f"all:{t}" for t in terms[:max_terms])


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _evidence_id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_arxiv_feed(xml_text: str, query: str) -> list[Evidence]:
    """Parse an arXiv Atom feed; exposed for offline tests."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    results: list[Evidence] = []
    for entry in root.findall("a:entry", _ATOM):
        title = _clean(entry.findtext("a:title", "", _ATOM))
        abstract = _clean(entry.findtext("a:summary", "", _ATOM))
        url = _clean(entry.findtext("a:id", "", _ATOM))
        if not (title and abstract and url.startswith("http")):
            continue
        url = url.replace("http://", "https://", 1)
        authors = [_clean(a.findtext("a:name", "", _ATOM)) for a in entry.findall("a:author", _ATOM)]
        published = _parse_date(entry.findtext("a:published", None, _ATOM))
        byline = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        year = f" ({published.year})" if published else ""
        results.append(
            Evidence(
                evidence_id=_evidence_id("arxiv", url),
                url=url,
                domain="arxiv.org",
                title=title,
                text=f"{title}. {byline}{year}. Abstract: {abstract}"[:3000],
                retrieval_timestamp=datetime.now(timezone.utc),
                publication_date=published,
                provider="arxiv",
                retrieval_metadata={"query": query, "authors": authors},
            )
        )
    return results


def parse_semantic_scholar(data: dict, query: str) -> list[Evidence]:
    """Parse a Semantic Scholar paper-search response; papers without abstracts are skipped."""
    results: list[Evidence] = []
    for paper in data.get("data") or []:
        title = _clean(paper.get("title") or "")
        abstract = _clean(paper.get("abstract") or "")
        if not title or not abstract:
            continue
        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
        url = paper.get("url") or ""
        if not url.startswith("http"):
            continue
        authors = [a.get("name", "") for a in paper.get("authors") or []]
        byline = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        year = paper.get("year")
        venue = paper.get("venue") or ""
        cites = paper.get("citationCount")
        meta = " ".join(
            part for part in (
                f"({year})" if year else "",
                f"Venue: {venue}." if venue else "",
                f"Citations: {cites}." if cites is not None else "",
            ) if part
        )
        results.append(
            Evidence(
                evidence_id=_evidence_id("s2", paper.get("paperId") or url),
                url=url,
                domain="semanticscholar.org",
                title=title,
                text=f"{title}. {byline} {meta} Abstract: {abstract}"[:3000],
                retrieval_timestamp=datetime.now(timezone.utc),
                publication_date=_parse_date(paper.get("publicationDate")),
                provider="semantic_scholar",
                retrieval_metadata={
                    "query": query, "citation_count": cites, "arxiv_id": arxiv_id, "authors": authors,
                },
            )
        )
    return results


async def _cached(cache: Optional[FileCache], key: str, fetch) -> list[Evidence]:
    if cache:
        hit = cache.get(key)
        if isinstance(hit, dict) and isinstance(hit.get("items"), list):
            return [Evidence(**e) for e in hit["items"]]
    results = await fetch()
    if cache and results:
        cache.set(key, {"items": [e.model_dump(mode="json") for e in results]})
    return results


async def search_arxiv(
    query: str, max_results: int = 3, timeout: int = 15, cache: Optional[FileCache] = None
) -> list[Evidence]:
    async def fetch() -> list[Evidence]:
        params = {"search_query": arxiv_query(query), "max_results": str(max_results), "sortBy": "relevance"}
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": _USER_AGENT},
                                     follow_redirects=True) as client:
            response = await client.get(ARXIV_ENDPOINT, params=params)
            response.raise_for_status()
        return parse_arxiv_feed(response.text, query)

    return await _cached(cache, f"arxiv:{hashlib.sha256(f'{query}|{max_results}'.encode()).hexdigest()}", fetch)


async def search_semantic_scholar(
    query: str, max_results: int = 3, timeout: int = 15, cache: Optional[FileCache] = None,
    api_key: Optional[str] = None,
) -> list[Evidence]:
    async def fetch() -> list[Evidence]:
        headers = {"User-Agent": _USER_AGENT}
        if api_key:
            headers["x-api-key"] = api_key
        # Over-fetch: many results lack abstracts and are skipped.
        params = {"query": query, "limit": str(max_results * 2), "fields": S2_FIELDS}
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(S2_ENDPOINT, params=params)
            if response.status_code == 429:
                # Keyless requests share one public pool; one short retry often gets through.
                await asyncio.sleep(1.5)
                response = await client.get(S2_ENDPOINT, params=params)
            response.raise_for_status()
        return parse_semantic_scholar(response.json(), query)[:max_results]

    return await _cached(cache, f"s2:{hashlib.sha256(f'{query}|{max_results}'.encode()).hexdigest()}", fetch)


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    """The same paper often comes back from both sources; keep the first copy (arXiv)."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in items:
        key = re.sub(r"[^a-z0-9]", "", ev.title.lower())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(ev)
    return out


async def search_papers(
    queries: list[str],
    max_results: int = 3,
    timeout: int = 15,
    cache: Optional[FileCache] = None,
    s2_api_key: Optional[str] = None,
    errors: Optional[list[str]] = None,
) -> list[Evidence]:
    """Query arXiv and Semantic Scholar for each query; a failing source is skipped, not fatal.

    Keyless Semantic Scholar shares a public rate limit and often answers 429, so
    each source failure is recorded in ``errors`` and the other source still counts.
    """
    tasks = []
    for q in queries:
        tasks.append(search_arxiv(q, max_results, timeout, cache))
        tasks.append(search_semantic_scholar(q, max_results, timeout, cache, s2_api_key))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    collected: list[Evidence] = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            if errors is not None:
                source = "arxiv" if i % 2 == 0 else "semantic_scholar"
                errors.append(f"{source}: {type(r).__name__}: {str(r)[:120]}")
            continue
        collected.extend(r)
    return _dedupe(collected)


__all__ = [
    "arxiv_query",
    "is_academic_question",
    "parse_arxiv_feed",
    "parse_semantic_scholar",
    "search_arxiv",
    "search_semantic_scholar",
    "search_papers",
]

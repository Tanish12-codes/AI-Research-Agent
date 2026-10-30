"""GitHub REST API tool for structured repository metadata."""
import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

from ..schemas import Evidence
from ..cache import FileCache

GITHUB_API_BASE = "https://api.github.com"
GITHUB_REPO_PATTERN = re.compile(r"(?:^|//|\.)github\.com/([\w.-]+)/([\w.-]+)")

NON_REPO_OWNERS: set[str] = {
    "about", "apps", "collections", "customer-stories", "enterprise", "events",
    "explore", "features", "login", "marketplace", "new", "notifications", "orgs",
    "pricing", "search", "security", "settings", "site", "sponsors", "topics", "trending",
}


def extract_github_repos(urls: list[str]) -> list[tuple[str, str]]:
    """Extract (owner, repo) pairs from URLs deterministically."""
    repos: list[tuple[str, str]] = []
    seen: set[str] = set()

    for url in urls:
        match = GITHUB_REPO_PATTERN.search(url)
        if match:
            owner = match.group(1)
            repo = match.group(2).removesuffix(".git")
            if owner.lower() in NON_REPO_OWNERS or not repo:
                continue
            key = f"{owner}/{repo}".lower()
            if key not in seen:
                seen.add(key)
                repos.append((owner, repo))

    return repos


async def fetch_github_repo_metadata(
    owner: str,
    repo: str,
    token: Optional[str] = None,
    timeout: int = 10,
    cache: Optional[FileCache] = None,
) -> Optional[Evidence]:
    """Fetch structured metadata for a GitHub repository."""
    cache_key = f"github:{owner}/{repo}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            payload = cached.get("value", cached) if (isinstance(cached, dict) and "value" in cached and "evidence_id" not in cached) else cached
            return Evidence(**payload)

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        text_parts = [
            f"Repository: {data.get('full_name', f'{owner}/{repo}')}",
            f"Description: {data.get('description', 'No description')}",
            f"Language: {data.get('language', 'Unknown')}",
            f"Stars: {data.get('stargazers_count', 0)}",
            f"Forks: {data.get('forks_count', 0)}",
            f"Open Issues: {data.get('open_issues_count', 0)}",
            f"License: {data.get('license', {}).get('name', 'Unknown') if data.get('license') else 'None'}",
            f"Created: {data.get('created_at', 'Unknown')}",
            f"Last Updated: {data.get('updated_at', 'Unknown')}",
            f"Last Pushed: {data.get('pushed_at', 'Unknown')}",
            f"Default Branch: {data.get('default_branch', 'main')}",
            f"Topics: {', '.join(data.get('topics', []))}",
            f"Archived: {data.get('archived', False)}",
        ]

        text = "\n".join(text_parts)

        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        evidence = Evidence(
            evidence_id=f"github-{hashlib.sha256(f'{owner}/{repo}'.encode()).hexdigest()[:12]}",
            url=data.get("html_url", f"https://github.com/{owner}/{repo}"),
            domain="github.com",
            title=data.get("full_name", f"{owner}/{repo}"),
            text=text,
            retrieval_timestamp=datetime.now(timezone.utc),
            publication_date=created_at,
            provider="github",
            retrieval_metadata={
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "language": data.get("language"),
                "archived": data.get("archived", False),
                "topics": data.get("topics", []),
            },
            relevance_score=None,
            authority_score=0.8,
        )

        if cache:
            cache.set(cache_key, evidence.model_dump(mode="json"))

        return evidence

    except (httpx.TimeoutException, httpx.HTTPStatusError, Exception):
        return None


async def fetch_multiple_repos(
    repos: list[tuple[str, str]],
    token: Optional[str] = None,
    timeout: int = 10,
    cache: Optional[FileCache] = None,
    max_concurrent: int = 3,
) -> list[Evidence]:
    """Fetch metadata for multiple repos with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(owner: str, repo: str) -> Optional[Evidence]:
        async with semaphore:
            return await fetch_github_repo_metadata(owner, repo, token, timeout, cache)

    tasks = [fetch_one(owner, repo) for owner, repo in repos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, Evidence)]

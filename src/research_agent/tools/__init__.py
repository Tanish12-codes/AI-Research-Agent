"""Research agent tools for web search, page fetching, and GitHub metadata."""

from .search import (
    SearchProvider,
    TavilySearchProvider,
    BraveSearchProvider,
    MockSearchProvider,
    DuckDuckGoSearchProvider,
    WikipediaSearchProvider,
    SearchWithFallback,
    build_search,
)
from .fetch import fetch_page
from .github import (
    extract_github_repos,
    fetch_github_repo_metadata,
    fetch_multiple_repos,
)

__all__ = [
    "SearchProvider",
    "TavilySearchProvider",
    "BraveSearchProvider",
    "MockSearchProvider",
    "DuckDuckGoSearchProvider",
    "WikipediaSearchProvider",
    "SearchWithFallback",
    "build_search",
    "fetch_page",
    "extract_github_repos",
    "fetch_github_repo_metadata",
    "fetch_multiple_repos",
]

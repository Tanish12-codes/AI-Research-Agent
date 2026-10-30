"""Evidence filtering, deduplication, and quality assessment."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .schemas import Evidence
from .verifier import _entity_in_text, _normalize_for_match

HIGH_AUTHORITY_DOMAINS: set[str] = {
    "arxiv.org",
    "semanticscholar.org",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "developer.mozilla.org",
    "en.wikipedia.org",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "microsoft.com",
    "google.com",
    "aws.amazon.com",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
    "nist.gov",
    "w3.org",
    "ietf.org",
    "rfc-editor.org",
}

MEDIUM_AUTHORITY_DOMAINS: set[str] = {
    "medium.com",
    "dev.to",
    "towardsdatascience.com",
    "realpython.com",
    "geeksforgeeks.org",
    "freecodecamp.org",
    "reddit.com",
    "news.ycombinator.com",
    "bbc.com",
    "reuters.com",
    "nytimes.com",
}

LOW_AUTHORITY_INDICATORS: list[str] = [
    "blogspot",
    "wordpress.com",
    "wix.com",
    "weebly.com",
    "tumblr.com",
    "livejournal.com",
]

TRACKING_PARAM_PREFIXES: tuple[str, ...] = (
    "utm_",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
    "mkt_tok",
    "yclid",
)

TRACKING_PARAM_EXACT: set[str] = {
    "ref",
    "ref_src",
    "source",
    "fb_ref",
    "fb_source",
}

STOP_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


def _extract_domain(url_or_domain: str) -> str:
    """Extract clean domain name without protocol, port, or www prefix."""
    if not url_or_domain:
        return ""
    if "://" in url_or_domain or url_or_domain.startswith("//"):
        try:
            parsed = urlparse(url_or_domain)
            netloc = parsed.netloc or parsed.path
        except Exception:
            netloc = url_or_domain
    else:
        netloc = url_or_domain.split("/")[0]

    host = netloc.split(":")[0].strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments, sorting query params, and stripping tracking."""
    if not url or not isinstance(url, str):
        return ""

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return url.strip()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_params = parse_qs(parsed.query, keep_blank_values=False)
    filtered_params: dict[str, list[str]] = {}

    for k, v in query_params.items():
        k_lower = k.lower()
        if k_lower in TRACKING_PARAM_EXACT:
            continue
        if any(k_lower.startswith(prefix) for prefix in TRACKING_PARAM_PREFIXES):
            continue
        filtered_params[k] = sorted(v)

    sorted_query = urlencode(sorted(filtered_params.items()), doseq=True)

    return urlunparse((scheme, netloc, path, "", sorted_query, ""))


def content_hash(text: str) -> str:
    """Generate SHA256 hash of normalized text (lowercase, collapsed whitespace)."""
    if not text:
        return hashlib.sha256(b"").hexdigest()
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_near_duplicate(
    evidence: Evidence, existing: list[Evidence], similarity_threshold: float = 0.85
) -> bool:
    """Check if evidence is a near-duplicate using content hash first, then Jaccard similarity."""
    if not existing:
        return False

    text = evidence.text or ""
    chash = evidence.content_hash or content_hash(text)
    norm_url = normalize_url(evidence.url) if evidence.url else ""

    tokens = set(re.findall(r"\b\w+\b", text.lower()))

    for item in existing:
        item_hash = item.content_hash or content_hash(item.text or "")
        if chash == item_hash:
            return True

        item_tokens = set(re.findall(r"\b\w+\b", (item.text or "").lower()))
        if not tokens and not item_tokens:
            return True
        if not tokens or not item_tokens:
            continue

        intersection = len(tokens.intersection(item_tokens))
        union = len(tokens.union(item_tokens))
        if union > 0:
            jaccard = intersection / union
            if jaccard >= similarity_threshold:
                return True

    return False


def compute_lexical_relevance(
    evidence_text: str, query: str, target_entities: list[str]
) -> float:
    """Score 0-1 based on query term overlap + entity presence + TF weighting."""
    if not evidence_text:
        return 0.0

    evidence_lower = evidence_text.lower()
    text_tokens = re.findall(r"\b[a-z0-9_-]+\b", evidence_lower)
    if not text_tokens:
        return 0.0

    counts: dict[str, int] = {}
    for token in text_tokens:
        counts[token] = counts.get(token, 0) + 1

    all_query_terms = re.findall(r"\b[a-z0-9_-]+\b", (query or "").lower())
    filtered_query_terms = [t for t in all_query_terms if t not in STOP_WORDS]
    query_terms = filtered_query_terms if filtered_query_terms else all_query_terms

    if query_terms:
        unique_terms = set(query_terms)
        matched_terms = [t for t in unique_terms if t in counts]
        overlap_ratio = len(matched_terms) / len(unique_terms)

        tf_sum = sum(math.log1p(counts[t]) for t in matched_terms)
        tf_max = len(unique_terms) * math.log1p(5)
        tf_score = min(1.0, tf_sum / tf_max) if tf_max > 0 else 0.0
        query_score = 0.6 * overlap_ratio + 0.4 * tf_score
    else:
        query_score = 0.5

    valid_entities = [e.strip() for e in target_entities if e and e.strip()]
    if valid_entities:
        matched_entities = sum(1 for e in valid_entities if e.lower() in evidence_lower)
        entity_score = matched_entities / len(valid_entities)
        final_score = 0.5 * query_score + 0.5 * entity_score
    else:
        final_score = query_score

    return min(1.0, max(0.0, round(final_score, 4)))


def compute_authority_score(domain: str) -> float:
    """0-1 score based on domain authority tiers."""
    clean = _extract_domain(domain)
    if not clean:
        return 0.3

    if any(ind in clean for ind in LOW_AUTHORITY_INDICATORS):
        return 0.3

    if clean in HIGH_AUTHORITY_DOMAINS or any(
        clean.endswith("." + d) for d in HIGH_AUTHORITY_DOMAINS
    ):
        return 0.95

    if clean.endswith(".gov") or clean.endswith(".edu") or clean.endswith(".mil"):
        return 0.90

    if clean in MEDIUM_AUTHORITY_DOMAINS or any(
        clean.endswith("." + d) for d in MEDIUM_AUTHORITY_DOMAINS
    ):
        return 0.70

    return 0.50


def extract_publication_date(text: str, metadata: dict[str, Any]) -> Optional[datetime]:
    """Try to extract publication date from metadata or text patterns."""
    date_keys = [
        "publication_date",
        "published_date",
        "date",
        "pubdate",
        "timestamp",
        "created_at",
        "publish_date",
        "date_published",
        "article:published_time",
    ]

    meta = metadata or {}
    for key in date_keys:
        val = meta.get(key)
        if not val:
            continue

        if isinstance(val, datetime):
            return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)

        if isinstance(val, (int, float)):
            try:
                ts = val / 1000.0 if val > 1e11 else float(val)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass

        if isinstance(val, str):
            val_clean = val.strip()
            try:
                dt = datetime.fromisoformat(val_clean.replace("Z", "+00:00"))
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass

            for fmt in (
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(val_clean, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

    if not text:
        return None

    iso_match = re.search(r"\b(20\d\d|19\d\d)[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
    if iso_match:
        try:
            year, month, day = (
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )
            return datetime(year, month, day, tzinfo=timezone.utc)
        except Exception:
            pass

    month_pattern = (
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(20\d\d|19\d\d)\b"
    )
    month_match = re.search(month_pattern, text, re.IGNORECASE)
    if month_match:
        month_str, day_str, year_str = month_match.group(1), month_match.group(2), month_match.group(3)
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                dt = datetime.strptime(f"{month_str} {day_str} {year_str}", fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass

    return None


def compute_recency_weight(pub_date: Optional[datetime], max_age_days: int = 365) -> float:
    """1.0 for recent, decaying for older. Returns 0.5 for neutral/unknown."""
    if pub_date is None:
        return 0.5

    if pub_date.tzinfo is None:
        dt = pub_date.replace(tzinfo=timezone.utc)
    else:
        dt = pub_date.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    age_days = (now - dt).total_seconds() / 86400.0

    if age_days <= 0:
        return 1.0

    half_life = max(1.0, max_age_days / 2.0)
    decay = 0.5 ** (age_days / half_life)
    weight = 0.2 + 0.8 * decay

    return min(1.0, max(0.2, round(weight, 4)))


def compute_diversity_bonus(evidence: Evidence, existing_domains: set[str]) -> float:
    """Bonus for new domains (promotes source diversity)."""
    raw_domain = getattr(evidence, "domain", None) or getattr(evidence, "url", "")
    domain = _extract_domain(raw_domain)

    if not domain:
        return 0.0

    if domain not in existing_domains:
        return 0.15

    return 0.0


def score_evidence(
    evidence: Evidence,
    query: str,
    target_entities: list[str],
    existing_evidence: list[Evidence],
    existing_domains: set[str],
) -> float:
    """Compute combined evidence score and populate evidence score attributes."""
    text = evidence.text or ""
    url = evidence.url or ""
    domain = evidence.domain or _extract_domain(url)

    rel_score = compute_lexical_relevance(text, query, target_entities)

    auth_score = compute_authority_score(domain or url)

    pub_date = evidence.publication_date
    if pub_date is None:
        pub_date = extract_publication_date(text, evidence.retrieval_metadata or {})
        if pub_date:
            try:
                evidence.publication_date = pub_date
            except Exception:
                pass

    recency_weight = compute_recency_weight(pub_date)

    diversity_bonus = compute_diversity_bonus(evidence, existing_domains)

    trust_level = getattr(evidence, "trust_level", 1.0)
    if evidence.is_quarantined:
        trust_level = 0.0

    base_score = (
        0.45 * rel_score
        + 0.25 * auth_score
        + 0.15 * recency_weight
        + 0.15 * diversity_bonus
    )
    final_score = min(1.0, max(0.0, base_score * trust_level))

    try:
        evidence.relevance_score = round(rel_score, 4)
        evidence.authority_score = round(auth_score, 4)
    except Exception:
        pass

    return round(final_score, 4)


def filter_and_rank_evidence(
    evidence_list: list[Evidence],
    query: str,
    target_entities: list[str],
    max_evidence: int = 15,
) -> list[Evidence]:
    """Full pipeline: deduplicate, score, rank, and budget."""
    if not evidence_list:
        return []

    unique_evidence: list[Evidence] = []
    for ev in evidence_list:
        if ev.is_quarantined:
            continue
        if not is_near_duplicate(ev, unique_evidence):
            unique_evidence.append(ev)

    scored_items: list[tuple[float, Evidence]] = []
    seen_domains: set[str] = set()

    for ev in unique_evidence:
        score = score_evidence(ev, query, target_entities, unique_evidence, seen_domains)
        scored_items.append((score, ev))
        dom = _extract_domain(ev.domain or ev.url)
        if dom:
            seen_domains.add(dom)

    scored_items.sort(key=lambda item: item[0], reverse=True)

    selected = [ev for _, ev in scored_items[:max_evidence]]
    return selected


def check_coverage(
    evidence: list[Evidence],
    required_entities: list[str],
    required_aspects: list[str],
) -> tuple[float, list[str], list[str]]:
    """Evaluate whether evidence adequately covers required entities and aspects."""
    total_required = [
        item.strip()
        for item in (required_entities + required_aspects)
        if item and item.strip()
    ]
    if not total_required:
        return 1.0, [], []

    corpus = " ".join(ev.text for ev in evidence if ev.text).lower()

    covered_items: list[str] = []
    missing_items: list[str] = []
    normalized_corpus = _normalize_for_match(corpus)

    for item in total_required:
        item_lower = item.lower()
        if item_lower in corpus:
            covered_items.append(item)
            continue

        words = [
            w for w in re.findall(r"\b\w+\b", item_lower)
            if len(w) > 2 and w not in STOP_WORDS
        ]
        if (words and all(w in corpus for w in words)) or _entity_in_text(item, normalized_corpus):
            covered_items.append(item)
        else:
            missing_items.append(item)

    coverage_ratio = len(covered_items) / len(total_required)
    return round(coverage_ratio, 4), covered_items, missing_items


__all__ = [
    "HIGH_AUTHORITY_DOMAINS",
    "MEDIUM_AUTHORITY_DOMAINS",
    "LOW_AUTHORITY_INDICATORS",
    "normalize_url",
    "content_hash",
    "is_near_duplicate",
    "compute_lexical_relevance",
    "compute_authority_score",
    "extract_publication_date",
    "compute_recency_weight",
    "compute_diversity_bonus",
    "score_evidence",
    "filter_and_rank_evidence",
    "check_coverage",
]

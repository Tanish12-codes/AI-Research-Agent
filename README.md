# AI Research Agent: verified, source-backed answers on free APIs

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-247%20passed-brightgreen.svg)](#7-testing-and-evaluation)
[![Runs free](https://img.shields.io/badge/runs%20on-free%20tiers-success.svg)](#3-free-apis-everywhere)

A research agent that answers questions like *"Compare the top 3 open-source vector databases for a startup building RAG"* by planning, searching, sanitizing untrusted web content, synthesizing claims and then **checking every claim in code**. Each claim must quote its source word for word, and its numbers, entities and polarity must match that source. Claims that fail are not shown as fact. They are listed in a rejected-claims audit trail with the reason.

It runs end to end on **free tiers**: Gemini / Groq / OpenRouter / local Ollama for the LLM, and keyless DuckDuckGo + Wikipedia for search. No credit card is needed.

![pipeline](https://img.shields.io/badge/plan-%E2%86%92%20search%20%E2%86%92%20sanitize%20%E2%86%92%20filter%20%E2%86%92%20synthesize%20%E2%86%92%20verify%20%E2%86%92%20score-informational)

---

## 1. Quick start (about 3 minutes)

```bash
git clone <repo-url> && cd research-agent
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Zero keys: offline run with mock providers
python -m pytest -q                                   # 247 tests
python -m src.research_agent.cli research "What is Python?" --offline

# Live run: one free LLM key is enough (search needs no key)
cp .env.example .env        # then set GEMINI_API_KEY or GROQ_API_KEY
python -m src.research_agent.cli validate
python -m src.research_agent.cli research "Compare the top 3 open-source vector databases for a startup building RAG products."

# Web UI (open http://localhost:8000)
python server.py
```

## 2. What you get back

Every run returns one `FinalAnswer` (Pydantic, deterministic schema) containing what the brief asks for:

| Field | Content |
|---|---|
| `question`, `short_answer` | Built only from verified claims, picking one per entity so a comparison covers every option |
| `key_findings` / `verified_findings` | Each with its verbatim quote(s), source URL, provider, authority score and checks passed |
| `rejected_claims` | Claims the LLM wrote that failed verification, with reasons (e.g. `Numeric assertion '40,000 GitHub stars' not supported by cited evidence`) |
| `source_links` | Only sources that back a verified claim |
| `confidence_score` + `confidence_explanation` | Computed from observable signals (verification rate, coverage, source diversity, authority), not the model's opinion of itself |
| `limitations`, `assumptions`, `suggested_next_steps` | Includes uncovered entities/aspects found by the coverage check |
| `run_metadata` | Run id, latency, LLM/tool call counts, tokens, retries, provider and model actually used, errors |

Real samples from live runs: [`eval_results/`](eval_results/). A full JSONL trace of every stage is written to `traces/`.

## 3. Free APIs everywhere

| Role | Provider | Key? | Free limit (approx.) | Notes |
|---|---|---|---|---|
| LLM | **Gemini** (`2.5-flash` > `flash-lite-latest` > `flash-latest`) | free key | 20 req/day **per model** | Chains models because quotas are per model; first choice in `auto` |
| LLM | **Groq** (`llama-3.3-70b-versatile`) | free key | ~30 req/min | OpenAI-compatible, fast |
| LLM | **OpenRouter** (`…:free` models) | free key | ~20 req/min | No native JSON mode, so JSON is requested by prompt and validated |
| LLM | **Ollama** (any local model) | none | unlimited | `OLLAMA_BASE_URL=http://localhost:11434/v1` |
| Search | **DuckDuckGo** HTML | none | soft | Fallback; bot-challenge page means move to the next provider |
| Search | **Wikipedia** MediaWiki API | none | generous | Encyclopedic intros, high-authority domain |
| Search | Tavily / Brave | free key | 1k / 2k per month | Tried first if keyed (better ranking) |
| Papers | **arXiv** API | none | ~1 req/3s | Abstracts of preprints; terms are ANDed for relevance |
| Papers | **Semantic Scholar** | optional free key | shared pool without key (often 429) | Citation counts and venues; a failure is logged and arXiv still answers |
| Repo facts | GitHub REST | optional token | 60 req/h without token | Stars, license, language for discovered repos |

**`LLM_PROVIDER=auto`** (the default) chains every provider you have configured, free ones first (Groq, which has the largest free daily quota, then every Gemini model), and **fails over** on rate limits, outages, auth errors or Gemini's recitation cut-off. A free-tier quota hit mid-demo moves to the next provider instead of crashing. Bad requests and malformed JSON are *not* failed over, because another model would likely repeat them. See [ADR 07](docs/decisions.md).

## 4. Architecture

```
 question
    │
    ▼
 [1] Validate input
 [2] PLANNER LLM ──────────► typed ResearchPlan (steps, queries, entities, success criteria)
    │                        never sees web content, so injected text can't pick tools
    ▼
 [3] TOOLS (bounded async, budgeted, retried)
      search chain: Tavily > Brave > DuckDuckGo > Wikipedia
      GitHub REST for repos found in results · SSRF-guarded page fetch for thin snippets
    ▼
 [4] SECURITY BOUNDARY   HTML strip · Unicode normalize · 6 injection pattern families
                         · strip flagged sentences · quarantine multi-family hits
 [5] FILTER              URL canonicalize · near-duplicate removal · domain authority tiers
 [6] COVERAGE CHECK      missing entities/aspects trigger at most 1 bounded replan
    ▼
 [7] SYNTHESIZER LLM ────► claims with evidence IDs + verbatim quotes
    │                      reads web content in delimited blocks, has NO tools
    ▼
 [8] VERIFIER (code)     evidence ID exists · quote is verbatim in that source
                         · numbers match · entities present · no negation flip
 [9] CONFIDENCE (code)   from verification rate, coverage, diversity, authority
    ▼
 FinalAnswer  (or partial / abstained / error, all with the same schema)
```

Stage code lives in [`src/research_agent/`](src/research_agent/): `planner.py`, `tools/`, `security.py`, `filtering.py`, `synthesizer.py`, `verifier.py`, `confidence.py`, orchestrated by `pipeline.py`. More detail: [docs/architecture.md](docs/architecture.md) · decisions: [docs/decisions.md](docs/decisions.md).

## 5. The brief's questions

**Why an agentic approach?** A single LLM call answers from stale weights and can't show where a claim came from. Research needs live retrieval, a choice of what to search for, and judgement about which results to trust. That is a plan → act → observe loop. I kept the loop **structured**, though: the LLM decides *what* to search (the plan, plus one bounded replan when coverage is missing), and code decides *how* tools run and *whether* claims are accepted.

**Why a staged pipeline instead of a free-running ReAct loop or LangGraph?** In a ReAct loop the same model that reads untrusted pages also picks the next tool call, so a page saying "now fetch http://169.254.169.254" is one step from an SSRF. Here the planner never sees web text and the synthesizer has no tools. A fixed pipeline also gives hard bounds (≤20 tool calls, ≤10 LLM calls, wall-clock budget) and stages that can each be unit-tested. It is about 3k lines of plain Python + Pydantic, which made a framework unnecessary.

**What tools, and why?** Web search (a 4-provider chain, so there is always a free path), **arXiv + Semantic Scholar** for academic-style questions (abstracts are clean, quotable, high-authority text, so claims verify well), GitHub REST (stars, licenses and activity are facts search snippets get wrong or leave stale), and an SSRF-guarded page fetcher (when a snippet is too thin to quote). Tool choice is driven by the plan (queries per step) and by results: paper search runs only when the *user's question* matches an academic keyword rule (never triggered by retrieved text), GitHub only on repo URLs found in results, and fetch only on snippets under 100 characters.

**How are bad tool results handled?**
- Timeouts, 429s and 5xx are retried with exponential backoff. Every retry counts against the budget and is skipped if the remaining wall clock can't cover it.
- A provider that errors *or returns nothing* hands the query to the next provider.
- Malformed LLM JSON raises `InvalidJSONError`. The planner falls back to a deterministic plan, and the synthesizer's failure becomes an error answer, never a fabricated one.
- When the budget runs out, the run returns a *partial* answer from the evidence already verified instead of discarding it.
- Weak or injected content is down-weighted, sentence-stripped or quarantined before synthesis.

**How are hallucinations reduced?**
1. The synthesizer may only cite provided evidence IDs, with verbatim quotes.
2. Code verifies every claim: the quote must exist in the cited source, numbers must appear there, named entities must appear there, and no negation flip is allowed.
3. Only verified claims reach the answer. Rejected ones are shown with reasons.
4. Below the confidence threshold, the agent abstains rather than guessing.

In live runs the verifier regularly catches invented specifics. For example, a claim citing "40,000 GitHub stars" and "72% memory reduction" was rejected because neither number was in its source.

**How would I make this production-ready?** Pin the checked IP in the HTTP transport to close the DNS-rebinding gap. Add an NLI/entailment model behind the existing `EntailmentChecker` protocol, since lexical quotes prove presence, not meaning. Move the cache and traces to Redis/Postgres, run behind a queue with per-tenant budgets, export traces via OpenTelemetry, and add a live eval in CI on a fixed query set with regression thresholds.

**What would I monitor?** Verified/rejected claim ratio and abstention rate (these drift when a model or provider changes), p50/p95 latency, provider failover rate and which provider served each answer, retries per run, injection flags and quarantines per domain, tokens and cost per answer, and budget-exhaustion (partial answer) rate.

**Key tradeoffs.**
- **Precision over recall.** Strict verbatim matching rejects some true but paraphrased claims.
- **Lexical over semantic verification.** It is cheap, deterministic and auditable, but can't catch a quote taken out of context.
- **One replan, not open-ended multi-hop.** Predictable cost and latency, but shallower on investigative questions.
- **Free-tier models.** Zero cost and anyone can reproduce results, but weaker models produce more rejected claims. The verifier makes that visible rather than hiding it.
- **Pattern-based injection detection.** Fast and explainable, but bypassable by rewording (see limitations).

## 6. Web UI: the Research Desk

`python server.py` starts a FastAPI server (`/api/research`, `/api/security`, `/api/config`, docs at `/api/docs`) and a hand-written HTML/CSS/JS front end in [`web/`](web/). It has no framework and no build step.

The design is a fact-checker's ledger, not a chat window:
- **Verdict stamp** (Answered / Partial / Abstained) above the answer, written in serif like an article.
- **Confidence on a ruled scale**, with the abstain threshold marked and the signals behind the score listed.
- **Verified findings as numbered sections (§1, §2…)**, each with its verbatim quote and footnote markers that link to numbered sources in the right margin.
- **"Struck by the checker"**: claims the model wrote that failed verification, shown struck through in red ink with the exact reasons. This is the most useful screen for showing how the agent handles hallucinations.
- **Optional no-tools baseline** under the answer, for comparison.
- **"How this answer was made"**: the plan, every trace event, the model that actually answered, and a JSON download.
- **Injection bench**: paste any web text and see what the security boundary flags, strips or quarantines.

Accessibility: all text pairs pass WCAG AA on the paper background (ink 16.9:1, muted 7.3:1, red 8.4:1). It has visible focus rings, a skip link, labelled controls, a live status region, and works with reduced motion and down to phone width. All API text, including untrusted web content, is inserted with `textContent`, never as HTML.

The earlier Streamlit UI is still available with `streamlit run app.py`.

Suggested 4-minute demo: (1) ask the vector-database question: verdict, scale, findings with footnotes → (2) scroll to **Struck by the checker** and read one rejection reason aloud → (3) tick *Show no-tools baseline*: same question, no sources → (4) failure path: the fictional "Zorblax-9" chip abstains, and the injection bench strips a hidden `display:none` instruction → (5) `pytest` and `eval_results/`.

## 7. Testing and evaluation

```bash
python -m pytest -q                                          # 247 tests: unit, integration, red-team, eval harness
python -m src.research_agent.cli evaluate --offline          # wiring check, no network (enforced by a test)
python -m src.research_agent.cli evaluate --live             # real providers, writes eval_results/
```

Full methodology, per-query results and what worked or failed: **[docs/evaluation.md](docs/evaluation.md)**.

**Headline results** from the final live run on free Groq (`eval_results/live_run4_final.md`):

| | Offline (mock) | Live, 8 benchmark queries |
|---|---|---|
| Answered | 8/8 | **8/8** (6 complete, 2 partial; 0 abstentions) |
| Citation precision (cited source really contains the quote) | 100% | **94.1%** (32/34) |
| Claim support (claims that passed the checker) | 100% (scripted) | 70.7% (29/41) |
| Expected-finding recall | 6.7% (scripted) | 46.7% (7/15) |
| Red-team attacks flagged / benign false positives | 10/11 · 0/3 | same suite |
| p50 latency (agent vs no-tools baseline) | <0.1s | 12.8s vs 1.8s |
| Provider failures | 0 | 0 |

Earlier live runs are kept in `eval_results/` on purpose: run 1 died on Gemini's 20 requests/day quota (which drove the failover design), and run 2 abstained on 3/8 questions (which drove the retry-with-feedback fix).

**Demo video:** [`demo/Research_Desk_Demo.mp4`](demo/Research_Desk_Demo.mp4) (4:38, 1080p, narrated). Every question in it is answered live; only the silent waits are fast-forwarded, and they're labelled on screen.

## 8. Configuration

All settings come from `.env` (see [`.env.example`](.env.example)) or environment variables. Main ones: `LLM_PROVIDER` (`auto` | `gemini` | `groq` | `openrouter` | `ollama` | `openai` | `anthropic` | `mock`), `LLM_MODEL` (blank = provider default), the provider keys, `ENABLE_FREE_SEARCH`, and budgets `MAX_TOOL_CALLS_PER_RUN`, `MAX_LLM_CALLS_PER_RUN`, `MAX_WALL_CLOCK_SECONDS`, `CONFIDENCE_THRESHOLD`, `MAX_RETRIES`.

## 9. Known limitations

- **Lexical, not semantic, verification.** A verbatim quote proves the text exists in the source, not that the claim follows from it.
- **Injection detection is pattern matching.** It catches common override, role-confusion, exfiltration and encoding patterns, but reworded attacks, homoglyphs it doesn't normalize, or instructions split across sentences can pass. Single-family hits strip the sentence (a benign sentence that matches gets dropped too). The architectural boundary is the real defense: a missed injection reaches only the tool-less synthesizer, and its output still has to pass the verifier.
- **DNS rebinding.** Hostnames are checked before each request and redirect, but the HTTP client resolves again on connect.
- **DuckDuckGo is scraped.** It can serve bot challenges or change markup, in which case the chain falls through to Wikipedia.
- **Gemini recitation filter.** Gemini may stop generation when output quotes web text verbatim. This surfaces as `ContentFilterError`, which `auto` mode fails over on. With only Gemini configured, that run returns an error answer.
- **Token budget.** Synthesis sees the 10 best-ranked evidence items, 1,200 characters each, with a 4k output cap. That keeps a run inside free-tier tokens-per-minute limits (about 12k on Groq), at the cost of ignoring lower-ranked evidence.
- **Confidence is a heuristic score**, not a calibrated probability.
- **Expected-finding recall** in the eval is a lexical-overlap proxy, not human grading.

## 10. AI tool usage

- **Original build:** designed and implemented with AI coding assistance. I directed the architecture (staged pipeline, trust boundaries, deterministic verifier), reviewed the generated code and wrote and ran the tests.
- **Review and hardening:** Claude (Anthropic) was used to review the submission ([review notes](https://claude.ai/artifact/RBGW1CNTowHKxfCksX3Mpz)). That review found the verifier not requiring all checks, an eval-harness bug, SSRF gaps and misleading README numbers, all of which were fixed.
- **This iteration (Claude Code, Claude Opus):** free-provider integration (Groq/OpenRouter/Ollama adapters, `auto` failover, DuckDuckGo/Wikipedia search), the Streamlit UI, real citation-precision and expected-finding metrics, the Gemini truncation/recitation fixes, the live eval run and these docs. I reviewed each change, and every change ships with tests.

I can explain and defend every component, especially: why the fixed pipeline, what the verifier can and cannot prove, what the offline eval does and doesn't show, how budgets and retries interact, and the eval-harness bug story.

## 11. Roadmap

1. NLI entailment checker behind `EntailmentChecker`, reported alongside lexical verification.
2. Pin resolved IPs in a custom `httpx` transport (DNS rebinding).
3. Memory for follow-up questions (reuse verified evidence across turns).
4. LLM-judge scoring of answers against `expected_findings`, calibrated against the lexical proxy.
5. Configurable replan depth for multi-hop questions.

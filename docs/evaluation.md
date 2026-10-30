# Evaluation

This document covers what was tested, how it is measured, the actual offline and live results, and what worked and what broke. Every number below comes from a file in [`eval_results/`](../eval_results/) or from `pytest`.

## 1. Three layers of evaluation

| Layer | Command | What it proves | What it does **not** prove |
|---|---|---|---|
| Unit / integration / red-team tests | `python -m pytest -q` (223 tests) | Each stage behaves as specified, including failure paths, failover, SSRF and injection handling | Answer quality |
| Offline eval (mock LLM + mock search) | `cli evaluate --offline` | The pipeline is wired end to end. It makes no network calls, which a test enforces | Answer quality: the mock LLM quotes mock evidence, so support rate is high by construction |
| Live eval (real LLM + real search) | `cli evaluate --live` and `cli research "<q>"` | Real behaviour: grounding, rejections, abstention, latency, provider failures | Correctness against human judgement (recall uses a lexical proxy) |

## 2. Metrics and formulas

All metrics keep explicit denominators (see `src/research_agent/evaluation/metrics.py`).

| Metric | Formula | Notes |
|---|---|---|
| Claim support rate | verified claims / all synthesized claims | Lower in live runs is expected: it is the verifier doing its job |
| **Citation precision** | cited sources that actually contain one of the claim's verbatim quotes / all (claim, source) citations | Rewritten in this iteration. It previously counted "has a URL and non-empty text", so it was always 100%. Now it penalizes over-citation |
| **Expected-finding recall** | golden `expected_findings` stated in the answer / all golden findings | New. Lexical proxy: at least 50% of the finding's content words appear in the verified claims or short answer. Undercounts paraphrases |
| Unsupported claims / answer | rejected claims / queries | |
| False abstention rate | abstained answerable queries / answerable queries | |
| Mean entity coverage | required entities named in the answer / required entities | |
| Attack success rate | unflagged attacks / attacks (11) | |
| Benign false-positive rate | quarantined benign controls / controls (3) | |
| p50/p95 latency, tokens, provider failures, retries | from `run_metadata` | |

## 3. Results

### 3.1 Offline (mock) run — `eval_results/eval_summary_mock.md`

| Metric | Value |
|---|---|
| Answerable queries answered | 8/8 (0% false abstention) |
| Claim support rate | 100% (8/8), by construction |
| Citation precision | 100% (8/8) |
| Expected-finding recall | **6.7% (1/15)**: correct and expected, since the mock answers are scripted wiring fixtures, not real answers. Reported to show the metric isn't rubber-stamping |
| Red-team: attacks flagged | 10/11 (`atk-002` hidden CSS is removed by the HTML sanitizer before detection, so the regex metric counts it as a miss) |
| Benign false positives | 0/3 |

### 3.2 Live run 1: single Gemini model — `eval_results/live_run1_single_model_quota_exhausted.md`

Config: `gemini-2.5-flash` only; search chain Tavily > Brave > DuckDuckGo > Wikipedia.

| Query | Status | Confidence | Verified / rejected |
|---|---|---|---|
| q-001 What is Python and who created it? | completed | 0.79 | 7 / 3 |
| q-002 Compare FastAPI and Flask for microservices | completed | 0.80 | 5 / 5 |
| q-003 … q-008 | **error** | 0 | `429 RESOURCE_EXHAUSTED` |
| Baseline (all 8) | error | — | same quota |

Aggregate over the 2 completed queries: claim support 60% (12/20), **citation precision 90.9% (20/22)**, 12 provider failures / 12 retries.

**What broke:** Gemini's free tier allows **20 requests per day per model**. Development runs had used most of it, so the eval died at query 3. Retries with backoff couldn't help, because the quota resets daily. This failure motivated the next round of changes:
- chain several Gemini models (each has its own quota) and fail over between them and other free providers;
- give each provider its own timeout, so an overloaded model (`gemini-flash-latest` returned 504 after 30s) can't use up the whole call budget;
- stick with the last provider that worked, so later calls don't re-hit an exhausted one;
- record the provider/model that actually answered in `run_metadata`, because under failover it differs from the config.

### 3.3 Live runs after the fixes (samples in `eval_results/`)

| # | Query | Expected behaviour | Result | Served by |
|---|---|---|---|---|
| A | Compare the top 3 open-source vector databases for a startup building RAG products | Comparison covering 3 DBs, sourced | **completed, 0.81**: 3 verified (Milvus, Chroma, Weaviate), **1 rejected**: "Qdrant raised $28 million", quote and number not in source → [sample](../eval_results/sample_live_comparison_vector_dbs.json) | gemini-flash-lite-latest (after 2.5-flash 429) |
| B | Research the pros and cons of a multi-agent architecture for customer support automation | Pros and cons with sources | **Before verifier fix: abstained, 0/8 verified.** After: **completed, 0.89, 6 verified, 1 rejected** (invented statistic "Only 1 in 10 agentic AI pilots…") → [sample](../eval_results/sample_live_pros_cons_multi_agent.json) | gemini-flash-lite-latest |
| C | What did the Zorblax-9 quantum database announce at its 2026 launch event? (fictional) | Must not invent an answer; abstain | **abstained, 0.00**: "Could not synthesize any claims from the evidence" → [sample](../eval_results/sample_live_abstention_fictional_entity.json) | gemini-flash-lite-latest |
| D | Compare the top 3 open-source vector databases (earlier, gemini-2.5-flash) | as A | completed, 0.40: 8 verified / 3 rejected. Rejections caught "40,000 GitHub stars", "72% memory reduction" and entities "Milvus 2.6", "RaBitQ" absent from sources | gemini-2.5-flash |
| E | Multi-agent pros/cons (earlier, gemini-2.5-flash) | as B | completed, 0.79 | gemini-2.5-flash |

### 3.3b Live run 2: full 8-query run on Groq after all fixes — `eval_results/live_run2_groq_full.md`

Config: `LLM_PROVIDER=auto` (served by `groq/openai/gpt-oss-120b`); search chain as above; arXiv on academic questions.

| Query | Status | Confidence | Verified / rejected |
|---|---|---|---|
| q-001 Python and its creator | completed | 0.69 | 1 / 2 |
| q-002 FastAPI vs Flask | completed | 0.76 | 3 / 3 |
| q-003 GraphQL vs REST pros/cons | **abstained** | 0.00 | 0 / 0 (no claims synthesized) |
| q-004 OpenTelemetry tracing in Python | completed | 0.88 | 9 / 1 |
| q-005 LLM agent frameworks with traction in 2024 | **abstained** | 0.05 | 0 / 3 |
| q-006 Rust borrow checker | completed | 0.89 | 3 / 1 |
| q-007 SQLite WAL trade-offs | **abstained** | 0.05 | 0 / 5 |
| q-008 PyTorch vs JAX | completed | 0.96 | 8 / 0 |
| Baseline (all 8) | completed | | no sources |

Aggregate: claim support 61.5% (24/39), **citation precision 100% (29/29)**, expected-finding recall 26.7% (4/15), false abstention **37.5% (3/8)**, p50 latency 16.2s (baseline 2.3s), 44.8k tokens for 8 queries (about 5.6k per query), **0 provider failures / 0 retries**.

**Reading it honestly:** every answer that was shown was fully grounded (100% citation precision), but the agent is too conservative. 3 answerable questions abstained. In q-005 and q-007 every claim failed the strict verbatim/entity checks; in q-003 the model returned no claims. That is the precision-over-recall tradeoff in practice. The next fixes would be a second synthesis attempt with feedback when all claims are rejected, and an NLI entailment check so faithful paraphrases can pass.

### 3.4 What worked

- **The verifier catches real hallucinations live.** In every completed live run it rejected claims with invented specifics (funding amounts, star counts, percentages, product versions), not just formatting slips. Those claims are listed with reasons rather than silently dropped.
- **Abstention on a fictional entity (C).** The agent returned nothing rather than a plausible-sounding launch summary.
- **Keyless search works.** Tested live in isolation, DuckDuckGo returned three relevant vector-database comparison articles and Wikipedia returned the "Vector database" article, with no key. In the full runs a Tavily key was present, so Tavily (first in the chain) served the searches. Keyless search is the fallback path, not what produced the numbers above.
- **Failover kept runs alive.** After the fixes, every live run reached a final answer (2 completed, 1 correct abstention) on the fallback model while the primary was exhausted, and the model that actually answered is recorded.
- **Citation precision of about 91% live** shows the synthesizer mostly cites sources that really contain its quotes. The remaining roughly 9% are over-citations (an extra source listed alongside the matching one).

### 3.5 What did not work (and what changed)

| Problem found | Root cause | Fix |
|---|---|---|
| Live eval died after 2 queries | Single model, 20 req/day free quota | Gemini model chain + cross-provider `auto` failover |
| Run hung 60s then errored with a blank message | Overloaded model hung; one timeout covered the whole chain; `str(TimeoutError())` is empty | Per-provider timeout inside the chain; error message falls back to the exception type |
| Synthesis JSON truncated (`finish_reason=length`) | 3000-token cap; 2.5 "thinking" tokens count toward it | Cap raised to 8000; thinking disabled on 2.5-flash; at most 10 claims with short quotes |
| `gemini-2.0-flash` returned 404 | Model retired | Default to `-latest` aliases plus a 2.5 fallback; 404 is a failover-able `ModelUnavailableError` |
| Multi-agent query abstained with 0/8 verified | Entity check required lowercase *concept* phrases verbatim ("multi-agent architecture") | Concept phrases match on word stems; **named entities still need an exact match** (tests prove "RaBitQ" / "Milvus 2.6" are still rejected) |
| Short answer only described the first product, lowercased names ("milvus") | Took the first two claims, `.lower()` on the second | Picks one verified claim per entity, keeps case |
| Citation precision always 100% | Measured "URL present" | Measures quote containment per cited source |

### 3.6 Known gaps in the evaluation

- **False abstention is 37.5% live.** See section 3.3b; the strict verifier over-rejects. A retry-with-feedback synthesis pass is the first next step.
- **Expected-finding recall is lexical.** An LLM judge calibrated on a few hand-labelled answers would be better.
- **Baseline comparison.** The no-tools baseline failed on quota in run 1. Single-query comparisons are available in the UI (toggle "Compare with no-tools baseline"). The structural difference holds regardless: the baseline has no sources and no verification.
- **Red-team suite is pattern-level.** It doesn't include paraphrased or homoglyph attacks, which are known to slip past regex detection (see README limitations).

## 4. Reproducing

```bash
python -m pytest -q
python -m src.research_agent.cli evaluate --offline
python -m src.research_agent.cli evaluate --live --max-queries 8     # needs at least one free LLM key
python -m src.research_agent.cli research "<question>"               # writes eval_results/research_<id>.json
```

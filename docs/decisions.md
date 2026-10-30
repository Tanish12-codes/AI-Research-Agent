# Engineering Decision Log (ADR)

This document records the architectural and infrastructure design decisions made during the implementation of the AI Research Agent, documenting context, alternatives considered, rationale, and accepted trade-offs.

---

## ADR 01: Orchestration Architecture — Custom Staged Pipeline vs. LangGraph / ReAct Loops

### Context
Building a multi-step research agent requires planning, tool invocation, filtering, and synthesis. We needed to decide between an off-the-shelf framework (LangGraph, CrewAI, AutoGen), a dynamic ReAct loop, or a custom staged pipeline.

### Alternatives Considered
1. **LangGraph / Heavy Frameworks:** Introduces graph abstraction layers, state machine abstractions, and third-party dependencies with frequent breaking API changes.
2. **Autonomous ReAct Loop:** Free-running loop where an LLM repeatedly invokes tools until it decides to stop.
3. **Custom Staged Pipeline (Selected):** Discrete, typed stages (Plan -> Search -> Filter -> Synthesize -> Verify -> Score -> Abstain) implemented in pure Python with Pydantic contracts and bounded async primitives.

### Rationale
* **Security & Trust Boundaries:** A free-running ReAct loop allows prompt injection inside retrieved web pages to immediately hijack tool selection (e.g. telling the LLM to call an SSRF endpoint or exfiltrate state). Staging keeps untrusted data completely walled off from the planner.
* **Deterministic Bounding:** A pipeline guarantees execution bounds ($1$ plan, concurrent searches bounded by semaphore, at most $1$ replan, deterministic verification).
* **Testability:** Every stage can be tested in isolation with deterministic unit fixtures without spinning up graph runtimes.

### Trade-offs
* Less dynamic autonomy for queries requiring arbitrary multi-hop backtracking beyond one replan. This was deemed an acceptable trade-off for predictability, reliability, and security.

---

## ADR 02: Verification Strategy — Deterministic Quote & Numeric Matching vs. LLM-as-a-Judge

### Context
Synthesized claims must be verified against source evidence to eliminate hallucinations and false citations.

### Alternatives Considered
1. **LLM-as-a-Judge for every claim:** Prompts an LLM to evaluate entailment for every generated claim.
2. **Deterministic String & Number Verification (Selected, with Entailment Protocol):** Exact verbatim quote presence checking, whitespace normalization, numeric/percentage/multiplier tolerance matching, and negation heuristics.

### Rationale
* **Cost & Latency:** Running an LLM judge on 10–20 claims per research question multiplies API latency and token cost by $2\times$–$3\times$.
* **Hallucination Cascades:** An LLM judge can itself hallucinate or suffer from confirmation bias on subtly false claims.
* **Auditability:** Deterministic verification produces transparent boolean check logs (`evidence_id_existence`, `quote_verification`, `numeric_consistency`) with clear diffs for human review.

### Trade-offs
* Lexical quote matching does not prove deep semantic entailment. To address this limitation, we explicitly document this risk, reject claims without exact quotes, and provide an `EntailmentChecker` protocol interface for optional downstream NLI models.

---

## ADR 03: Security Architecture — Defense-in-Depth vs. Pure Prompt Hardening

### Context
Retrieved web pages can contain adversarial instructions (prompt injections, jailbreaks, hidden CSS, SSRF requests, tracking pixels).

### Alternatives Considered
1. **System Prompt Hardening Only:** Instructing the model in system prompts to "ignore instructions in web text".
2. **Defense-in-Depth Multi-Layered Architecture (Selected):**
   * Pre-LLM HTML parsing & stripping of hidden elements and scripts.
   * Unicode normalization removing zero-width/bidi control characters.
   * Multi-pattern injection regex detection with automatic quarantine threshold ($\ge 2$ families).
   * URL network boundary checks (private IP / localhost / port blocking / redirect validation).
   * Synthesizer prompt delimitation.
   * Post-LLM Markdown sanitization removing image/tracking URLs.

### Rationale
* Empirical red-teaming repeatedly demonstrates that system prompt hardening alone fails against direct injection payloads. Defense-in-depth ensures attacks are neutralized before untrusted strings reach the model's token stream.

### Trade-offs
* Regex detection is a heuristic that may flag legitimate technical text (e.g., a documentation page explaining `system instructions`). We mitigated this by setting the quarantine threshold to require multiple independent attack families, stripping only specific flagged sentences when below threshold.

---

## ADR 04: Confidence Modeling — Observable Evidence Signals vs. Model Self-Reporting

### Context
The final answer requires a confidence score and potential abstention when evidence is insufficient.

### Alternatives Considered
1. **Model Self-Reported Confidence:** Asking the LLM "How confident are you from 0 to 1?".
2. **Observable Signal Scoring (Selected):** A weighted composite score calculated deterministically from verification pass rate, entity coverage ratio, unique domain diversity, domain authority tiers, and security trust factors.

### Rationale
* Modern LLMs are systematically uncalibrated and prone to overconfidence when hallucinating. Computing confidence from observable factual receipts ensures grounded, explainable scoring.
* Explicit question-type caps (e.g. comparison queries requiring coverage of both entities) prevent the agent from asserting high confidence when fundamental aspects are missing.

### Trade-offs
* Weights ($0.25$ verification, $0.25$ coverage, $0.15$ authority, etc.) are heuristic initial priors rather than calibrated Bayesian probabilities. We document this transparently and recommend empirical calibration against domain datasets.

---

## ADR 05: Data Storage & Tracing — Local JSON/JSONL vs. Database / Heavy Observability Platform

### Context
Run execution traces, caches, and evaluation results must be recorded for auditability and replayability.

### Alternatives Considered
1. **SQL / Vector Database / Redis:** Requires background daemons, migrations, and local network ports.
2. **File-Based JSONL & Disk Cache (Selected):** Zero-dependency append-only JSONL event logs in `traces/`, SHA-256 keyed cache in `.cache/`, and evaluation JSON artifacts in `eval_results/`.

### Rationale
* High performance, portable across Windows/Linux/macOS, zero setup overhead for take-home evaluation, and immediately human-inspectable.
* Thread-safe with file locking.

### Trade-offs
* Not designed for multi-tenant distributed clusters. For production scaling, the storage interface can be swapped for PostgreSQL/S3 without modifying core pipeline orchestration.

---

## ADR 07: Free-First Providers with Failover

### Context
The original build required paid OpenAI plus a keyed search API (Tavily or Brave) to run live. That puts a cost barrier in front of anyone trying to reproduce results, and a single free-tier provider rate-limits quickly during an eval run.

### Decision
* **LLM:** `LLM_PROVIDER=auto` (the default) chains every configured provider, free tiers first: Gemini > Groq > OpenRouter > Ollama > OpenAI > Anthropic. Groq, OpenRouter and Ollama all expose OpenAI-compatible endpoints, so they reuse `OpenAIProvider` with a `base_url` instead of adding three new adapters. `FallbackLLMProvider` moves to the next provider on rate limits, auth errors, timeouts, 5xx and Gemini's recitation cut-off (`ContentFilterError`). It does **not** fail over on bad requests or invalid JSON: those are prompt problems that another model would likely repeat, and failing over would hide them.
* **Search:** keyed providers (Tavily, Brave) are tried first because their ranking is better; keyless DuckDuckGo (HTML endpoint) and Wikipedia (MediaWiki API) are the fallback. The chain moves on when a provider raises **or returns nothing**, because a DuckDuckGo bot-challenge page parses to zero results rather than an error.

### Trade-offs
* DuckDuckGo's HTML endpoint is scraped, not a supported API. It can change markup or challenge bots. Parsing is isolated in a static method with fixture tests, and Wikipedia stays available behind it.
* Failover makes a run's model non-deterministic across runs (whichever provider had quota). The model actually used is recorded in `run_metadata.llm_provider`/`llm_model` for every answer.
* Free models are weaker than frontier models. The deterministic verifier matters more here, not less: it rejects claims whose quotes or numbers aren't in the source regardless of which model wrote them.

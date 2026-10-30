# Research Agent Architecture & Trust Boundaries

## 1. System Overview

The **Production-Minded AI Research Agent** is a multi-stage agentic information retrieval, synthesis, and verification system designed to answer open-ended and factual research questions with guaranteed evidence grounding, strict security isolation, and deterministic verification.

```
+-----------------------------------------------------------------------------+
|                               USER QUESTION                                 |
+-----------------------------------------------------------------------------+
                                       |
                                       v
                     [ 1. Validation & Input Sanitization ]
                                       |
                                       v
                      [ 2. Deterministic Planning LLM ]
                   (Zero external text or tools visible)
                                       |
                                       v
               +-----------------------------------------------+
               | 3. Concurrent Search & Discovered Repo Fetch  |
               |     (Tavily / Brave / GitHub REST API)        |
               +-----------------------------------------------+
                                       |
                                       v
                     [ 4. Security Sanitization & Isolation ]
                     - HTML strip (styles, hidden, script tags)
                     - Unicode NFC & Zero-Width / Bidi removal
                     - Prompt Injection detection (6 families)
                     - Quarantine Multi-indicator Threshold
                     - Strict SSRF / Private IP / Port guards
                                       |
                                       v
                    [ 5. Deduplication & Evidence Ranking ]
                    - URL normalization (tracking params stripped)
                    - Content hash + Jaccard near-duplicate filter
                    - Domain authority & recency weighting
                    - Evidence budget cap
                                       |
                                       v
                    [ 6. Plan Aspect Coverage & Replan ]
                    (At most 1 bounded replan on coverage gaps)
                                       |
                                       v
                     [ 7. Delimited Synthesis LLM ]
                     - Untrusted evidence in XML/delimiters
                     - No tools, network, or URL selection access
                     - Outputs claims with mandatory quotes & IDs
                                       |
                                       v
                   [ 8. Deterministic Claim Verification ]
                   - Evidence ID existence
                   - Verbatim quote exact normalized substring
                   - Number & multiplier tolerance checks
                   - Negation flip detection
                   - Entity co-occurrence matching
                                       |
                                       v
                    [ 9. Multi-Signal Confidence Scoring ]
                    - Observability-grounded signals (no self-eval)
                    - Question-type caps (comparison, factual, etc.)
                                       |
                                       v
                     [ 10. Policy Abstention & Output ]
                    (Abstain if zero verified findings or low conf)
                                       |
                                       v
                      +---------------------------------+
                      | FinalAnswer with Evidence Audit |
                      |    (Verified vs Rejected)       |
                      +---------------------------------+
```

---

## 2. Trust Boundaries & Threat Model

In LLM agent architectures, untrusted web content is a primary attack vector. The architecture establishes strict, non-porous trust boundaries:

### Boundary 1: The Planner (Zero Untrusted Data)
* **Input:** User query only.
* **Capabilities:** Categorizes question type, produces search queries, required entities, and measurable success criteria.
* **Isolation:** The planner is **never** exposed to raw search results, scraped web text, or external tools. It cannot be influenced by web-based prompt injections.

### Boundary 2: Retrieval Tools & SSRF Defense
* **HTTP Fetcher:**
  * Scheme restricted to `http` / `https`.
  * Ports restricted to `{80, 443, 8080, 8443}`.
  * Private, loopback, link-local, and reserved IP ranges (RFC 1918, RFC 3927, IMDS `169.254.169.254`, `localhost`, `127.0.0.1`, `::1`) are deterministically blocked prior to connection.
  * Manual step-by-step redirect validation (max 3 redirects) with protocol downgrade prevention (HTTPS -> HTTP blocked).
  * Hard response size ceiling (2MB max download) to prevent zip-bomb / memory exhaustion DoS.
* **GitHub Tool:**
  * Evaluates **only** repositories deterministically discovered in search results matching `github.com/<owner>/<repo>`.
  * The planner is prohibited from injecting arbitrary hallucinated repository endpoints.

### Boundary 3: Evidence Sanitization & Quarantine
* **HTML Stripping:** Removes `<script>`, `<style>`, `<iframe>`, `<object>`, `<embed>`, comments, and CSS-hidden containers (`display:none`, `visibility:hidden`, `aria-hidden="true"`).
* **Unicode Normalization:** Applies Unicode NFC normalization and strips zero-width spaces (`\u200b`, `\u200c`, `\u200d`, `\ufeff`) and bidirectional control overrides (`\u202a`–`\u202e`) used for visual deception.
* **Injection Scanning:** Evaluates content against 6 regex pattern families:
  1. `instruction_override`
  2. `role_manipulation`
  3. `system_prompt_extraction`
  4. `hidden_instructions`
  5. `data_exfiltration`
  6. `encoding_evasion`
* **Quarantine Policy:** If $\ge 2$ independent attack families are identified, the entire page is quarantined. If $< 2$, flagged sentences are removed, and the item's trust factor is penalized.

### Boundary 4: The Synthesizer (Data-Only Sandbox)
* **Access:** The synthesizer LLM receives untrusted evidence escaped and wrapped between explicit `=== BEGIN EVIDENCE [id] ===` and `=== END EVIDENCE [id] ===` delimiters.
* **Capabilities:** The synthesizer has **no tools, no API access, no ability to fetch URLs, and cannot write to external state**.
* **Output Sanitization:** Markdown image links (`![alt](url)`) and unapproved external URLs are stripped from the synthesized text to prevent CSS/image token exfiltration attacks.

---

## 3. Deterministic Verification Pipeline

The verification phase does not rely on an LLM "judge" prompt. It executes deterministic string and numerical algorithms:

1. **Evidence ID Existence:** Ensures every cited ID exists in the retrieved evidence set.
2. **Verbatim Quote Matching:** Normalizes whitespace and casing; verifies that the quoted passage exists verbatim within the cited evidence text.
3. **Numeric Consistency:** Normalizes numbers, percentages (e.g. `50%`), multipliers (`3x`), SI/shorthand units (`100k` -> `100,000`, `2.5M` -> `2,500,000`), and fractions (`3/4` -> `0.75`). Asserts numeric claims against evidence numbers within a 1% floating-point tolerance.
4. **Negation Flip Detection:** Heuristically detects negation polarity mismatches between the generated claim and cited quote (e.g., "Python is not interpreted" vs "Python is interpreted").
5. **Entity Co-occurrence:** Confirms that claimed entities appear in both the assertion and source text.

Failed checks partition the claim into `RejectedClaim` with explicit rejection reasons, creating a verifiable audit trail.

---

## 4. Confidence Scoring & Abstention

Confidence is computed from observable evidence signals rather than model self-reported certainty:

$$\text{Confidence} = 0.25 \cdot R_{\text{verify}} + 0.25 \cdot C_{\text{coverage}} + 0.15 \cdot D_{\text{domain}} + 0.15 \cdot A_{\text{authority}} + 0.10 \cdot S_{\text{checks}} + 0.10 \cdot T_{\text{trust}}$$

* **Question Type Adjustments:**
  * **Comparison:** Caps confidence at $0.50$ if any required comparison entity is missing.
  * **Pros/Cons:** Caps confidence at $0.50$ if only positive or only negative signals are discovered.
  * **Factual:** Penalizes confidence if average domain authority is below $0.50$.
* **Abstention Policy:** If verified claims count $= 0$, evidence count $= 0$, or $\text{Confidence} < \text{threshold}$ (default $0.30$), the agent abstains from giving a speculative answer and returns an explicit abstention rationale.

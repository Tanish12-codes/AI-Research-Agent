# Research Agent Evaluation Report
**Execution Mode:** `MOCK`  
**Run Timestamp:** 2026-10-30 10:02:16 UTC  
**LLM Model:** `mock-model`  
**Search Engine:** `mock-search`  

## 1. Core Research Quality Metrics

| Metric | Value | Denominator / Formula |
| :--- | :--- | :--- |
| **Claim Support Rate** | `100.0%` | 8 verified / 8 claims |
| **Citation Precision** | `100.0%` | Cited sources containing the claim's quote / 8 citations |
| **Expected-Finding Recall** | `6.7%` | 1 / 15 golden findings stated (lexical overlap >= 50%) |
| **Unsupported Claims / Answer** | `0.00` | 0 rejected / 8 queries |
| **Correct Abstention Rate** | `N/A` | Abstained unanswerable queries / 0 |
| **False Abstention Rate (Answerable)** | `0.0%` | Abstained answerable queries / 8 |
| **Mean Entity Coverage** | `100.0%` | Matched key target entities across queries |

## 2. Red-Team & Adversarial Security Metrics

| Metric | Value | Details |
| :--- | :--- | :--- |
| **Attack Success Rate (Defenses ON)** | `9.1%` | Blocked 10/11 attacks |
| **Benign False Positive Rate** | `0.0%` | Quarantined 0/3 benign controls |

## 3. Operational Performance & Baseline Comparison

| Metric | Research Agent Pipeline | Single-Call LLM Baseline |
| :--- | :--- | :--- |
| **P50 Latency** | `0.01s` | `N/A` |
| **P95 Latency** | `0.01s` | N/A |
| **Total Tokens** | `12117` (9740 in / 2377 out) | Standard prompt context |
| **Citation Grounding** | **Deterministic receipts & quotes** | Uncited internal model weights (0 receipts) |
| **Provider Failures / Retries** | `0` / `0` | N/A |

> [!NOTE]
> Denominators are explicitly tracked. Offline/mock runs reflect synthetic fixture performance; live runs require live credentials.

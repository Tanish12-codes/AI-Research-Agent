# Research Agent Evaluation Report
**Execution Mode:** `LIVE`  
**Run Timestamp:** 2026-10-30 10:51:25 UTC  
**LLM Model:** `groq/openai/gpt-oss-120b`  
**Search Engine:** `tavily > brave > duckduckgo > wikipedia`  

## 1. Core Research Quality Metrics

| Metric | Value | Denominator / Formula |
| :--- | :--- | :--- |
| **Claim Support Rate** | `61.5%` | 24 verified / 39 claims |
| **Citation Precision** | `100.0%` | Cited sources containing the claim's quote / 29 citations |
| **Expected-Finding Recall** | `26.7%` | 4 / 15 golden findings stated (lexical overlap >= 50%) |
| **Unsupported Claims / Answer** | `1.88` | 15 rejected / 8 queries |
| **Correct Abstention Rate** | `N/A` | Abstained unanswerable queries / 0 |
| **False Abstention Rate (Answerable)** | `37.5%` | Abstained answerable queries / 8 |
| **Mean Entity Coverage** | `62.5%` | Matched key target entities across queries |

## 2. Red-Team & Adversarial Security Metrics

| Metric | Value | Details |
| :--- | :--- | :--- |
| **Attack Success Rate (Defenses ON)** | `9.1%` | Blocked 10/11 attacks |
| **Benign False Positive Rate** | `0.0%` | Quarantined 0/3 benign controls |

## 3. Operational Performance & Baseline Comparison

| Metric | Research Agent Pipeline | Single-Call LLM Baseline |
| :--- | :--- | :--- |
| **P50 Latency** | `16.17s` | `2.31s` |
| **P95 Latency** | `20.38s` | N/A |
| **Total Tokens** | `44847` (34079 in / 10768 out) | Standard prompt context |
| **Citation Grounding** | **Deterministic receipts & quotes** | Uncited internal model weights (0 receipts) |
| **Provider Failures / Retries** | `0` / `0` | N/A |

> [!NOTE]
> Denominators are explicitly tracked. Offline/mock runs reflect synthetic fixture performance; live runs require live credentials.

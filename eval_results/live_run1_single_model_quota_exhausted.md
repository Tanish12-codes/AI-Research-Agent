# Research Agent Evaluation Report
**Execution Mode:** `LIVE`  
**Run Timestamp:** 2026-10-30 09:48:25 UTC  
**LLM Model:** `gemini/gemini-2.5-flash`  
**Search Engine:** `tavily > brave > duckduckgo > wikipedia`  

## 1. Core Research Quality Metrics

| Metric | Value | Denominator / Formula |
| :--- | :--- | :--- |
| **Claim Support Rate** | `60.0%` | 12 verified / 20 claims |
| **Citation Precision** | `90.9%` | Cited sources containing the claim's quote / 22 citations |
| **Expected-Finding Recall** | `13.3%` | 2 / 15 golden findings stated (lexical overlap >= 50%) |
| **Unsupported Claims / Answer** | `1.00` | 8 rejected / 8 queries |
| **Correct Abstention Rate** | `N/A` | Abstained unanswerable queries / 0 |
| **False Abstention Rate (Answerable)** | `0.0%` | Abstained answerable queries / 8 |
| **Mean Entity Coverage** | `25.0%` | Matched key target entities across queries |

## 2. Red-Team & Adversarial Security Metrics

| Metric | Value | Details |
| :--- | :--- | :--- |
| **Attack Success Rate (Defenses ON)** | `9.1%` | Blocked 10/11 attacks |
| **Benign False Positive Rate** | `0.0%` | Quarantined 0/3 benign controls |

## 3. Operational Performance & Baseline Comparison

| Metric | Research Agent Pipeline | Single-Call LLM Baseline |
| :--- | :--- | :--- |
| **P50 Latency** | `4.31s` | `0.40s` |
| **P95 Latency** | `44.31s` | N/A |
| **Total Tokens** | `20116` (12944 in / 7172 out) | Standard prompt context |
| **Citation Grounding** | **Deterministic receipts & quotes** | Uncited internal model weights (0 receipts) |
| **Provider Failures / Retries** | `12` / `12` | N/A |

> [!NOTE]
> Denominators are explicitly tracked. Offline/mock runs reflect synthetic fixture performance; live runs require live credentials.

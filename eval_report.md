# RAG Eval Report

Config: k=4, alpha=0.5, generation_model=llama-3.1-8b-instant, judge_model=llama-3.3-70b-versatile, n_questions=22

## Retrieval metrics (dense vs hybrid)

| mode | hit-rate@k | MRR | precision@k | recall@k |
|---|---|---|---|---|
| dense | 1.000 | 0.936 | 0.511 | 1.000 |
| hybrid | 1.000 | 0.924 | 0.455 | 1.000 |

## Generation metrics (hybrid path, LLM-as-judge)

- Mean faithfulness: 4.77/5
- Mean relevancy: 4.68/5
- Guardrail-flagged answers: 1/22 (4.5%)

## Human calibration

Skipped (--skip-calibration).

# RAG Eval Report

Config: k=4, alpha=0.5, generation_model=llama-3.1-8b-instant, judge_model=llama-3.3-70b-versatile, n_questions=22

## Retrieval metrics (dense vs hybrid)

| mode | hit-rate@k | MRR | precision@k | recall@k |
|---|---|---|---|---|
| dense | 0.955 | 0.864 | 0.511 | 0.955 |
| hybrid | 1.000 | 0.966 | 0.500 | 1.000 |
| hybrid+rerank | 1.000 | 1.000 | 0.489 | 1.000 |

## Generation metrics (hybrid path, LLM-as-judge)

- Mean faithfulness: 5.00/5
- Mean relevancy: 4.59/5
- Guardrail-flagged answers: 0/22 (0.0%)

## Human calibration

Skipped (--skip-calibration).

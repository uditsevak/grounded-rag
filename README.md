# Grounded — RAG Document Intelligence

Ask a question against an indexed document set and **audit how the answer was
assembled**: ranked sources, a dense-vs-sparse-vs-hybrid retrieval switch, and a
faithfulness reading that flags claims not grounded in the retrieved context.

Built on FAISS + BM25 + LangChain, generation and judging on Groq (free tier),
embeddings local via fastembed (ONNX — no torch, so it runs on free hosts). No
OpenAI key required.

> **Live demo:** _<add your Hugging Face Space URL here after the first deploy>_

<!-- add a screenshot: save one to docs/screenshot.png and uncomment -->
<!-- ![Grounded UI](docs/screenshot.png) -->


## What's actually in here

- **Hybrid retrieval** — dense (FAISS) + sparse (BM25) fused with reciprocal
  rank fusion, configurable weight `alpha`. Dense-only and sparse-only paths
  stay available so you can A/B them live in the UI.
- **Eval harness** (`eval.py`) — hit-rate@k, MRR, precision/recall for dense
  vs hybrid; LLM-as-judge faithfulness + answer relevancy; a human-vs-judge
  calibration step. Writes `eval_report.md` / `.json` from a real run.
- **Faithfulness guardrail** — every answer is scored for groundedness by the
  judge and flagged when it scores below threshold *or* the judge lists any
  unsupported claim. If the judge is rate-limited it degrades gracefully — the
  answer still returns, marked "check unavailable", rather than failing.
- **Bring your own document** — upload a PDF/TXT/MD and query it live. Each
  upload builds a throwaway in-memory retriever (`uploads.py`); it's never
  merged into the shared demo index and is evicted after a cap, so one visitor
  can't pollute another's corpus. Text-only (no OCR on uploads).
- **PDF + image ingestion** — `ingest.py` extracts PDF text and OCRs embedded
  diagrams (Tesseract) so image content is retrievable too.
- **Custom FastAPI + vanilla-JS frontend** — no framework, no build step.

## Eval results (real run, k=4, alpha=0.5, 22 questions)

| mode | hit-rate@k | MRR | precision@k | recall@k |
|---|---|---|---|---|
| dense | 1.000 | 0.936 | 0.511 | 1.000 |
| hybrid | 1.000 | 0.947 | 0.455 | 1.000 |

Mean faithfulness 4.86/5, mean relevancy 4.68/5, 1/22 answers guardrail-flagged
(a genuine caught hallucination — the model hedged into a claim the context
directly contradicts). Full detail in `eval_report.json`.

Honest reading: on this small, cleanly-separated corpus the two are close —
hybrid edges dense on MRR (0.947 vs 0.936) but dense wins on precision, since
dense alone already hits perfect recall and mixing in BM25 reshuffles the tail.
The point of the harness is that it *measures* this so you can tune `alpha` per
corpus, not that hybrid always wins. Bigger, noisier corpora with exact-keyword
queries (error codes, API paths) are where BM25 pulls more weight.

(This report was generated with the 70b judge; the deployed demo now defaults to
the 8b judge to stretch the free-tier token budget — re-run `python eval.py` to
regenerate with whatever `GROQ_JUDGE_MODEL` you set.)

## Run it locally

```bash
pip install -r requirements.txt        # Tesseract needed only to re-ingest: brew install tesseract
cp .env.example .env                    # add your GROQ_API_KEY (free: console.groq.com)
uvicorn server:app --port 7860          # open http://localhost:7860
```

The FAISS index and BM25 corpus are committed, so the app runs without a rebuild.
To rebuild from the PDFs in `data/`: `python ingest.py`.

Run the eval harness:

```bash
python eval.py                     # full run incl. interactive human calibration
python eval.py --skip-calibration  # metrics only, no prompts
```

## Layout

| file | role |
|---|---|
| `server.py` | FastAPI app: validated `/api/ask` + `/api/upload`, serves the frontend |
| `static/` | the frontend (index.html, style.css, app.js) |
| `uploads.py` | ephemeral per-user corpora for uploaded documents |
| `retrieval.py` | `HybridRetriever`: dense / sparse / hybrid (RRF) |
| `rag.py` | retrieve → generate → guardrail-check |
| `judge.py` | LLM-as-judge faithfulness + relevancy (fixed 0–5 rubric) |
| `guardrail.py` | flags answers below the faithfulness threshold |
| `providers.py` | model config (Groq chat/judge, local embeddings) |
| `ingest.py` / `vision.py` | build the index; OCR embedded PDF images |
| `eval.py` | retrieval + generation metrics, calibration, report |
| `golden_set.json` | 22 labeled Q&A with expected source docs |
| `test_metrics.py` | offline self-check for the metric math |

## Not production — deliberate scope

- Guardrail annotates rather than blocks flagged answers.
- Image handling is OCR, not image understanding — reads rendered text, won't
  describe a photo with no text.
- FAISS is a flat index over ~20 chunks (no IVF/HNSW/sharding).
- The public demo runs on one shared free-tier Groq key. Calls go through a
  small concurrency gate + retry/backoff, and the guardrail degrades gracefully
  when throttled, but heavy simultaneous traffic will still queue or skip the
  faithfulness check. No auth, no monitoring.

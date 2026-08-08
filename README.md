# Grounded — RAG Document Intelligence

**Ask a question against a document set and audit how the answer was assembled** —
ranked sources, a live dense/sparse/hybrid retrieval switch, and a faithfulness
score that flags any claim not grounded in the retrieved context.

Most RAG demos are a chat box that hands you an answer and asks you to trust it.
Grounded is built around the opposite idea: every answer comes with its evidence
and a groundedness check you can see.

### 🔗 [Try the live demo →](https://grounded-rag.onrender.com)

*Hosted free on Render — if it's been idle a few minutes the first load takes
~40s to wake, then it's fast. Runs entirely on free-tier infrastructure (Groq +
local embeddings), no paid APIs.*

<!-- Recommended: add a screenshot here. Save one to docs/screenshot.png (the
     flagged credit-card answer makes a strong hero shot) and uncomment: -->
<!-- ![Grounded UI](docs/screenshot.png) -->

---

## What this project demonstrates

- **Hybrid retrieval done properly** — dense (FAISS) + sparse (BM25) fused with
  reciprocal rank fusion, with a tunable weight. Dense-only and sparse-only paths
  stay live so you can A/B them in the UI and see when each wins.
- **A real evaluation harness**, not vibes — hit-rate@k, MRR, precision/recall
  against a labelled golden set, plus LLM-as-judge scoring for faithfulness and
  answer relevancy, plus a human-vs-judge calibration step. Numbers come from
  actually running it (`eval.py`), never hand-waved.
- **A hallucination guardrail** on the generation output — flags answers that
  aren't traceable to the retrieved context, and **degrades gracefully** under
  rate limits instead of failing the request.
- **Ingestion that handles real documents** — PDF text *and* OCR'd text from
  embedded diagrams, so image content is retrievable too.
- **Bring-your-own-document** — upload a PDF/TXT/MD and query it live against an
  ephemeral, per-user index that never touches the shared demo corpus.
- **A hand-built frontend** — custom FastAPI backend + vanilla HTML/CSS/JS, no
  framework, no build step.

## How it works

```
          ingest (PDF text + OCR'd diagrams) → chunk → embed → FAISS + BM25 index
                                                                      │
  question ─► hybrid retrieve (dense ⊕ sparse, RRF) ─► top-k chunks ──┤
                                                                      ▼
                             generate answer (grounded prompt) ─► LLM-as-judge
                                                                      │
                              faithfulness guardrail ◄────────────────┘
                                        │
                            answer + ranked sources + groundedness score
```

## Evaluation results

Real run over a 22-question labelled golden set (`k=4`, `alpha=0.5`):

| mode   | hit-rate@k | MRR   | precision@k | recall@k |
|--------|-----------:|------:|------------:|---------:|
| dense  | 1.000      | 0.936 | 0.511       | 1.000    |
| hybrid | 1.000      | 0.947 | 0.455       | 1.000    |

Mean faithfulness **4.86/5**, mean relevancy **4.68/5**, and **1/22** answers
guardrail-flagged — a genuine caught hallucination where the model hedged into a
claim the source contradicts.

**The honest read:** on this small, cleanly-separated corpus hybrid and dense are
close — hybrid edges dense on MRR while dense wins on precision, because dense
alone already hits perfect recall and mixing in BM25 just reshuffles the tail.
The value isn't "hybrid always wins" — it's that the harness *measures* the
tradeoff so you can tune per corpus. Bigger, noisier corpora with exact-keyword
queries (error codes, API paths) are where sparse retrieval earns its place.

## Engineering decisions worth calling out

This runs end-to-end on free infrastructure, which forced some deliberate calls:

- **fastembed (ONNX) instead of torch** for embeddings — same MiniLM model,
  ~200 MB instead of ~1 GB, so it fits a free 512 MB host and starts fast.
- **Smaller judge model + smarter flag logic** — the guardrail flags on a low
  score *or* any unsupported claim the judge names, which keeps detection
  reliable on a lightweight model and stretches the shared token budget.
- **Graceful degradation** — a rate-limited judge returns the answer marked
  "check unavailable" rather than dropping a successful generation; all model
  calls go through a concurrency gate with retry/backoff.
- **Reproducible builds** — dependencies are pinned to a set verified by a clean
  install + a 28-check integration test, so the deploy can't drift into a broken
  release.

## Run it locally

```bash
pip install -r requirements.txt        # Tesseract only needed to re-ingest: brew install tesseract
cp .env.example .env                    # add a free GROQ_API_KEY (console.groq.com)
uvicorn server:app --port 7860          # → http://localhost:7860
```

The FAISS index and BM25 corpus are committed, so it runs without a rebuild
(`python ingest.py` rebuilds from the PDFs in `data/`).

```bash
python eval.py --skip-calibration   # regenerate the metrics report
python test_metrics.py              # offline unit checks for the metric math
```

## Tech stack

Python · FastAPI · LangChain · FAISS · rank-bm25 · fastembed (ONNX) ·
Groq (Llama 3.1) · Tesseract OCR · vanilla JS · Docker / Render

## Scope & limitations

Deliberately a focused demo, not a production system:

- The guardrail annotates flagged answers rather than blocking them.
- Image handling is OCR (rendered text), not visual understanding.
- FAISS is a flat index over a small corpus — no ANN indexing or sharding.
- The demo shares one free-tier Groq key: a concurrency gate, retries, and
  graceful degradation keep normal traffic healthy, but a heavy burst will queue
  or skip the faithfulness check. No auth or monitoring.

*(The committed eval report was generated with a larger judge model; the live
demo defaults to a smaller one to stretch the free token budget. Re-run
`eval.py` with any `GROQ_JUDGE_MODEL` to regenerate.)*

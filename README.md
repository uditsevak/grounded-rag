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

- **A full retrieval stack** — **semantic chunking** (split on meaning, not fixed
  length) → **hybrid retrieval** (dense FAISS ⊕ sparse BM25, fused with reciprocal
  rank fusion) → an optional **cross-encoder reranker** → optional **LLM query
  rewriting**. Each stage is toggleable so you can measure its contribution.
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
   ingest → semantic chunk → embed → FAISS (dense) + BM25 (sparse) index
                                                          │
  question ─►[optional rewrite]─► hybrid retrieve (RRF) ─►│─► [optional rerank] ─► top-k
                                                                                     │
                              generate answer (grounded prompt) ◄────────────────────┘
                                        │
                              LLM-as-judge faithfulness guardrail
                                        │
                            answer + ranked sources + groundedness score
```

## Evaluation results

Real run over a 22-question labelled golden set (`k=4`, `alpha=0.5`), showing each
retrieval stage's contribution:

| retrieval        | hit-rate@k | MRR       | precision@k | recall@k |
|------------------|-----------:|----------:|------------:|---------:|
| dense only       | 0.955      | 0.864     | 0.511       | 0.955    |
| + sparse (hybrid)| 1.000      | 0.966     | 0.500       | 1.000    |
| + reranker       | 1.000      | **1.000** | 0.489       | 1.000    |

Generation (hybrid path): mean faithfulness **5.00/5**, mean relevancy **4.59/5**,
0 answers flagged — every answer was traceable to its sources.

**The honest read:** there's a real, measured progression — sparse retrieval lifts
hit-rate from 0.955 to 1.000 (BM25 catches exact-term matches like error codes
that dense embeddings miss), and the reranker lifts MRR to a perfect 1.000 (the
correct chunk lands at rank 1 for every question). Precision dips slightly because
each question usually has one relevant chunk but `k=4`. The point of the harness
is that it *measures* each stage so the additions are justified, not cargo-culted.

*(The guardrail shows 0 flags here because every generated answer was faithful;
its catch is demonstrable live — the "credit card" sample question makes the
model hedge into an unsupported claim, and the guardrail flags it.)*

## Engineering decisions worth calling out

This runs end-to-end on free infrastructure, which forced some deliberate calls:

- **fastembed (ONNX) instead of torch** for embeddings *and* reranking — same
  MiniLM models, ~200 MB instead of ~1 GB, so the whole stack (including the
  cross-encoder) fits a free 512 MB host and starts fast.
- **Two-tier judging** — the live per-request guardrail runs the fast 8B model
  (and flags on a low score *or* any unsupported claim it names, which stays
  reliable on a small model); the offline eval uses the stronger 70B judge for
  more credible scoring. Cheap where it's hot, accurate where it can afford to be.
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
# regenerate the report (70B judge reproduces the numbers above; omit it for the fast 8B judge)
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile python eval.py --skip-calibration
python test_metrics.py              # offline unit checks for the metric math
```

## Tech stack

Python · FastAPI · LangChain · FAISS · rank-bm25 · fastembed (ONNX embeddings +
cross-encoder reranker) · semantic chunking · Groq (Llama 3.1 / 3.3) ·
Tesseract OCR · vanilla JS · Render

## Scope & limitations

Deliberately a focused demo, not a production system:

- The guardrail annotates flagged answers rather than blocking them.
- Image handling is OCR (rendered text), not visual understanding.
- FAISS is a flat index over a small corpus — no ANN indexing or sharding.
- The demo shares one free-tier Groq key: a concurrency gate, retries, and
  graceful degradation keep normal traffic healthy, but a heavy burst will queue
  or skip the faithfulness check. No auth or monitoring.

- Query rewriting and reranking are opt-in per request (they add an LLM call /
  a second model load), off by default to protect the shared free-tier budget.

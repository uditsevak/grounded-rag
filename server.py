"""FastAPI app serving the RAG pipeline behind a small JSON API and the static UI.

One retriever is loaded at startup and shared across requests. The /ask handler
is deliberately a plain `def` (not `async`) so Starlette runs it in a worker
thread — the Groq and embedding calls block, and we don't want them stalling the
event loop.
"""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag import answer as rag_answer
from retrieval import HybridRetriever
from uploads import CorpusStore, extract_chunks

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(title="RAG Document Intelligence")
retriever = HybridRetriever()   # the shared demo corpus
custom_corpora = CorpusStore()  # ephemeral per-upload retrievers


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|sparse)$")
    k: int = Field(default=4, ge=1, le=10)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank: bool = False
    rewrite: bool = False
    corpus_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be blank")
        return v


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    # read one byte past the cap so we can tell "exactly at limit" from "over"
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large — max 5 MB.")
    try:
        doc_id, ids, texts, metadatas = extract_chunks(file.filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        corpus_id = custom_corpora.add(HybridRetriever.from_corpus(ids, texts, metadatas))
    except Exception:
        raise HTTPException(status_code=500, detail="Couldn't index that document.")

    return {"corpus_id": corpus_id, "filename": doc_id, "chunks": len(texts)}


@app.post("/api/ask")
def ask(req: AskRequest):
    active = retriever
    if req.corpus_id:
        active = custom_corpora.get(req.corpus_id)
        if active is None:
            raise HTTPException(status_code=404, detail="That uploaded document expired — upload it again.")

    try:
        result = rag_answer(
            req.question, active, mode=req.mode, k=req.k, alpha=req.alpha,
            rerank=req.rerank, rewrite=req.rewrite,
        )
    except Exception:
        # don't leak provider stack traces to a public endpoint
        raise HTTPException(status_code=502, detail="The model backend failed. Try again.")

    return {
        "answer": result["raw_answer"],
        "flagged": result["flagged"],
        "faithfulness": result["faithfulness"],
        "mode": req.mode,
        "k": req.k,
        "alpha": req.alpha,
        "reranked": result["reranked"],
        "rewritten_query": result["rewritten_query"],
        "sources": [
            {
                "rank": i + 1,
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "type": c["type"],
                "page": c.get("page"),
                "text": c["text"],
            }
            for i, c in enumerate(result["chunks"])
        ],
    }


# mounted last so /api/* wins; html=True serves index.html at /
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

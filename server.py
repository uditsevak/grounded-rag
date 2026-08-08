"""FastAPI app serving the RAG pipeline behind a small JSON API and the static UI.

One retriever is loaded at startup and shared across requests. The /ask handler
is deliberately a plain `def` (not `async`) so Starlette runs it in a worker
thread — the Groq and embedding calls block, and we don't want them stalling the
event loop.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag import answer as rag_answer
from retrieval import HybridRetriever

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="RAG Document Intelligence")
retriever = HybridRetriever()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|sparse)$")
    k: int = Field(default=4, ge=1, le=10)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be blank")
        return v


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        result = rag_answer(req.question, retriever, mode=req.mode, k=req.k, alpha=req.alpha)
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

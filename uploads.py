"""Ephemeral per-user corpora. A user uploads a document, we chunk it, build a
throwaway in-memory retriever, and query against that instead of the shared demo
index. Nothing is written to disk and the committed index is never touched.

Uploads are text-only: PDF text + .txt/.md. No OCR here (unlike ingest.py), so an
image-only PDF yields no text and is rejected — keeps the deploy image lean.
"""
import uuid
from collections import OrderedDict
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from retrieval import HybridRetriever

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNKS = 300   # bound embedding cost + memory for one upload
MAX_CORPORA = 20   # ponytail: FIFO cap on live custom corpora; add TTL/per-session if abused

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def extract_chunks(filename, data):
    """Return (doc_id, ids, texts, metadatas) from an uploaded file. Raises
    ValueError on an unsupported type or when no text can be extracted."""
    base = (filename or "document").rsplit("/", 1)[-1]
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    doc_id = (base.rsplit(".", 1)[0] or "document").strip() or "document"

    if ext == "pdf":
        reader = PdfReader(BytesIO(data))
        raw = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext in ("txt", "md", "markdown"):
        raw = data.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported file type — upload a PDF, TXT, or MD file.")

    chunks = _splitter.split_text(raw)
    if not chunks:
        raise ValueError("No text could be extracted from that file.")

    ids, texts, metadatas = [], [], []
    for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
        ids.append(f"{doc_id}::{i}")
        texts.append(chunk)
        metadatas.append({"doc_id": doc_id, "chunk_index": i, "type": "text"})
    return doc_id, ids, texts, metadatas


class CorpusStore:
    """In-memory map of corpus_id -> retriever with FIFO eviction. Not persisted;
    a Space restart clears it, and uploads past the cap drop the oldest."""

    def __init__(self, max_corpora=MAX_CORPORA):
        self._store = OrderedDict()
        self._max = max_corpora

    def add(self, retriever):
        corpus_id = uuid.uuid4().hex
        self._store[corpus_id] = retriever
        while len(self._store) > self._max:
            self._store.popitem(last=False)
        return corpus_id

    def get(self, corpus_id):
        return self._store.get(corpus_id)

"""Chunk data/*.md and data/*.pdf (text + embedded images), embed locally, and
build a FAISS index and a BM25 corpus.

PDF images are OCR'd (vision.py) and indexed as their own chunk, so
diagram/screenshot text becomes searchable alongside the surrounding page
text — not silently dropped.
"""
import os
import pickle
import re
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from providers import get_embeddings
from vision import caption_image

DATA_DIR = Path(__file__).parent / "data"
FAISS_DIR = Path(__file__).parent / "faiss_index"
BM25_CORPUS_PATH = Path(__file__).parent / "bm25_corpus.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# "recursive" (fixed-size, the default) or "semantic" (embed sentences, cut at
# similarity troughs). Semantic is implemented and eval'd, but on this short,
# factual corpus it grouped similar plans together and buried specific facts
# (e.g. the Business-plan SLA), so recursive retrieves better here. Set
# CHUNK_STRATEGY=semantic to compare.
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "recursive")
SEMANTIC_BREAKPOINT_PCTL = 30   # break where consecutive-sentence similarity is this low
SEMANTIC_MAX_CHARS = 700        # hard cap so a coherent run doesn't grow unbounded

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _semantic_split(text):
    """Split on meaning, not length: embed each sentence and start a new chunk
    where the similarity to the next sentence drops into the bottom percentile."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) < 3:
        return [text.strip()] if text.strip() else []

    vecs = np.array(get_embeddings().embed_documents(sentences))
    sims = [
        float(vecs[i] @ vecs[i + 1] / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i + 1]) + 1e-9))
        for i in range(len(sentences) - 1)
    ]
    cut = np.percentile(sims, SEMANTIC_BREAKPOINT_PCTL)

    chunks, cur = [], [sentences[0]]
    for i, sentence in enumerate(sentences[1:]):
        joined = " ".join(cur)
        if (sims[i] <= cut and len(joined) > 150) or len(joined) > SEMANTIC_MAX_CHARS:
            chunks.append(joined)
            cur = [sentence]
        else:
            cur.append(sentence)
    chunks.append(" ".join(cur))
    return chunks


def _split_text(text):
    if CHUNK_STRATEGY == "semantic":
        return _semantic_split(text)
    return _splitter.split_text(text)


def _chunk_markdown(path):
    doc_id = path.stem
    for i, chunk in enumerate(_split_text(path.read_text())):
        yield f"{doc_id}::{i}", chunk, {"doc_id": doc_id, "chunk_index": i, "type": "text"}


def _chunk_pdf(path):
    doc_id = path.stem
    reader = PdfReader(str(path))
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in _split_text(text):
            yield (
                f"{doc_id}::{chunk_index}",
                chunk,
                {"doc_id": doc_id, "chunk_index": chunk_index, "type": "text", "page": page_num},
            )
            chunk_index += 1

        for image in page.images:
            print(f"  captioning embedded image on {doc_id} p{page_num}: {image.name}")
            caption = caption_image(image.data)
            yield (
                f"{doc_id}::{chunk_index}",
                f"[Image on page {page_num}: {caption}]",
                {"doc_id": doc_id, "chunk_index": chunk_index, "type": "image", "page": page_num},
            )
            chunk_index += 1


def load_and_chunk():
    ids, texts, metadatas = [], [], []
    paths = sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.pdf"))
    for path in paths:
        chunker = _chunk_markdown if path.suffix == ".md" else _chunk_pdf
        print(f"chunking {path.name}...")
        for chunk_id, text, meta in chunker(path):
            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(meta)
    return ids, texts, metadatas


def build_index():
    ids, texts, metadatas = load_and_chunk()
    if not texts:
        raise RuntimeError(f"No documents found in {DATA_DIR}")

    vectorstore = FAISS.from_texts(texts, get_embeddings(), metadatas=metadatas, ids=ids)
    vectorstore.save_local(str(FAISS_DIR))

    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump({"ids": ids, "texts": texts, "metadatas": metadatas}, f)

    n_images = sum(1 for m in metadatas if m["type"] == "image")
    print(
        f"Indexed {len(texts)} chunks ({n_images} from images) "
        f"from {len(set(m['doc_id'] for m in metadatas))} docs "
        f"[{CHUNK_STRATEGY} chunking]"
    )
    print(f"FAISS index -> {FAISS_DIR}")
    print(f"BM25 corpus -> {BM25_CORPUS_PATH}")


if __name__ == "__main__":
    build_index()

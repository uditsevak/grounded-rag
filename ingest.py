"""Chunk data/*.md and data/*.pdf (text + embedded images), embed locally, and
build a FAISS index and a BM25 corpus.

PDF images are OCR'd (vision.py) and indexed as their own chunk, so
diagram/screenshot text becomes searchable alongside the surrounding page
text — not silently dropped.
"""
import pickle
from pathlib import Path

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

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _chunk_markdown(path):
    doc_id = path.stem
    for i, chunk in enumerate(_splitter.split_text(path.read_text())):
        yield f"{doc_id}::{i}", chunk, {"doc_id": doc_id, "chunk_index": i, "type": "text"}


def _chunk_pdf(path):
    doc_id = path.stem
    reader = PdfReader(str(path))
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in _splitter.split_text(text):
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
        f"from {len(set(m['doc_id'] for m in metadatas))} docs"
    )
    print(f"FAISS index -> {FAISS_DIR}")
    print(f"BM25 corpus -> {BM25_CORPUS_PATH}")


if __name__ == "__main__":
    build_index()

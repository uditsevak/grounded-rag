"""Dense (FAISS), sparse (BM25), and hybrid (RRF) retrieval over the ingested corpus."""
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from providers import get_embeddings

FAISS_DIR = Path(__file__).parent / "faiss_index"
BM25_CORPUS_PATH = Path(__file__).parent / "bm25_corpus.pkl"

# how many candidates each retriever contributes to the fusion pool, before
# cutting down to the requested top-k. Wider pool = fusion has more to work with.
FUSION_POOL_SIZE = 20

# standard RRF smoothing constant (Cormack et al.) — dampens the impact of rank 1.
RRF_K = 60

# Optional cross-encoder reranker (fastembed ONNX, ~80MB, no torch). When on, we
# retrieve a wider pool then re-score each (query, chunk) pair and keep the top-k.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_POOL_SIZE = 15


class HybridRetriever:
    def __init__(self):
        """Load the committed demo index from disk."""
        vectorstore = FAISS.load_local(
            str(FAISS_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
        with open(BM25_CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        self._build(vectorstore, corpus["ids"], corpus["texts"], corpus["metadatas"])

    @classmethod
    def from_corpus(cls, ids, texts, metadatas):
        """Build an in-memory retriever from a freshly chunked document (uploads)."""
        self = cls.__new__(cls)
        vectorstore = FAISS.from_texts(texts, get_embeddings(), metadatas=metadatas, ids=ids)
        self._build(vectorstore, ids, texts, metadatas)
        return self

    def _build(self, vectorstore, ids, texts, metadatas):
        self.vectorstore = vectorstore
        self.ids = ids
        self.texts = texts
        self.metadatas = metadatas
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.id_to_chunk = {
            cid: {"chunk_id": cid, "text": text, **meta}
            for cid, text, meta in zip(ids, texts, metadatas)
        }
        self._reranker = None  # loaded lazily on first rerank

    def _get_reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        return self._reranker

    def rerank(self, query, chunks, k):
        if not chunks:
            return chunks
        scores = list(self._get_reranker().rerank(query, [c["text"] for c in chunks]))
        ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)[:k]
        return [{**c, "rerank_score": float(s)} for c, s in ranked]

    def retrieve_dense(self, query, k=4):
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in results:
            chunk_id = doc.metadata["doc_id"] + "::" + str(doc.metadata["chunk_index"])
            out.append({**self.id_to_chunk[chunk_id], "score": float(score)})
        return out

    def retrieve_sparse(self, query, k=4):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.id_to_chunk[self.ids[i]], "score": float(scores[i])} for i in ranked]

    def retrieve_hybrid(self, query, k=4, alpha=0.5, pool_size=FUSION_POOL_SIZE, rrf_k=RRF_K):
        """alpha: weight on dense retrieval's rank contribution; (1-alpha) on sparse."""
        dense_pool = self.retrieve_dense(query, k=pool_size)
        sparse_pool = self.retrieve_sparse(query, k=pool_size)

        dense_rank = {c["chunk_id"]: r for r, c in enumerate(dense_pool)}
        sparse_rank = {c["chunk_id"]: r for r, c in enumerate(sparse_pool)}

        all_ids = set(dense_rank) | set(sparse_rank)
        fused_scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in dense_rank:
                score += alpha * (1.0 / (rrf_k + dense_rank[cid] + 1))
            if cid in sparse_rank:
                score += (1 - alpha) * (1.0 / (rrf_k + sparse_rank[cid] + 1))
            fused_scores[cid] = score

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:k]
        return [{**self.id_to_chunk[cid], "score": fused_scores[cid]} for cid in ranked_ids]

    def retrieve(self, query, k=4, mode="hybrid", alpha=0.5, rerank=False):
        # when reranking, pull a wider pool first so the cross-encoder has more to
        # re-score, then cut to k.
        pool_k = RERANK_POOL_SIZE if rerank else k
        if mode == "dense":
            hits = self.retrieve_dense(query, k=pool_k)
        elif mode == "sparse":
            hits = self.retrieve_sparse(query, k=pool_k)
        elif mode == "hybrid":
            hits = self.retrieve_hybrid(query, k=pool_k, alpha=alpha)
        else:
            raise ValueError(f"unknown mode: {mode}")

        if rerank:
            hits = self.rerank(query, hits, k)
        return hits

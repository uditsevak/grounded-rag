"""End-to-end query pipeline: retrieve -> generate -> guardrail-check."""
from guardrail import apply_guardrail
from providers import get_chat_llm, groq_invoke
from query_rewrite import rewrite_query
from retrieval import HybridRetriever

PROMPT_TEMPLATE = """Answer the question using only the context below. If the
context doesn't contain the answer, say you don't know — do not use outside
knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_context(chunks):
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def answer(question, retriever: HybridRetriever, mode="hybrid", k=4, alpha=0.5,
           rerank=False, rewrite=False):
    # optionally rewrite the query for retrieval only — the answer is still
    # generated against the user's original question.
    search_query = rewrite_query(question) if rewrite else question
    chunks = retriever.retrieve(search_query, k=k, mode=mode, alpha=alpha, rerank=rerank)
    context = build_context(chunks)
    llm = get_chat_llm()
    raw_answer = groq_invoke(llm, PROMPT_TEMPLATE.format(context=context, question=question)).content

    guarded = apply_guardrail(question, raw_answer, context)

    return {
        "question": question,
        "raw_answer": raw_answer,
        "answer": guarded["annotated_answer"],
        "flagged": guarded["flagged"],
        "faithfulness": guarded["faithfulness"],
        "sources": [c["chunk_id"] for c in chunks],
        "context": context,
        "chunks": chunks,
        "rewritten_query": search_query if rewrite else None,
        "reranked": rerank,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the uptime SLA on the Business plan?"
    retriever = HybridRetriever()
    result = answer(q, retriever)
    print(f"Q: {q}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources: {result['sources']}")

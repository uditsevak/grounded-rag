"""Optional query optimizer: rewrite a user's question into a search-friendly
query before retrieval (expand abbreviations, surface key terms, drop filler).
One cheap Groq call; used only when enabled so it doesn't spend tokens by default.
"""
from providers import get_chat_llm, groq_invoke

REWRITE_PROMPT = """Rewrite the question below into a concise search query for a
keyword + vector search over product documentation. Keep the important nouns and
any codes/numbers, expand obvious abbreviations, and drop conversational filler.
Return ONLY the rewritten query, no preamble.

Question: {question}
Search query:"""


def rewrite_query(question: str) -> str:
    out = groq_invoke(get_chat_llm(), REWRITE_PROMPT.format(question=question)).content.strip()
    # fall back to the original if the model returns something empty/degenerate
    return out or question

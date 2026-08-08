"""LLM-as-judge scoring: faithfulness (groundedness) and answer relevancy.

Fixed 0-5 rubric, structured output via a Pydantic schema so scores are
always parseable. Shared by eval.py (bulk scoring) and guardrail.py
(per-answer check before returning a response to the user).
"""
from pydantic import BaseModel, Field

from providers import get_judge_llm, groq_invoke

# Note on json_mode below: Groq's Llama models sometimes emit tool calls as
# literal text (`<function=...>{...}</function>`) instead of through the real
# tool-calling channel, which Groq rejects as tool_use_failed. json_mode
# sidesteps that path entirely (plain JSON in the message content). groq_invoke
# adds the shared concurrency gate + retry/backoff.

FAITHFULNESS_RUBRIC = """You are grading whether an ANSWER is faithful to the
provided CONTEXT (i.e. grounded — every factual claim in the answer is
supported by the context, with no invention or unsupported extrapolation).

Score 0-5:
5 = every claim is directly supported by the context.
4 = grounded; at most one minor claim is a reasonable inference, not a fabrication.
3 = mostly grounded, but one claim is not supported by the context.
2 = partially grounded; multiple claims are not supported by the context.
1 = mostly unsupported; only a small part of the answer is grounded.
0 = fabricated or contradicts the context entirely.

List any unsupported claims verbatim as they appear in the answer. If none, return an empty list."""

RELEVANCY_RUBRIC = """You are grading whether an ANSWER actually addresses the
QUESTION asked (independent of whether it's factually correct or grounded).

Score 0-5:
5 = directly and completely answers the question.
4 = answers the question with minor omissions.
3 = partially answers the question.
2 = tangentially related but mostly misses the question.
1 = barely related to the question.
0 = does not address the question at all."""


class FaithfulnessResult(BaseModel):
    score: int = Field(ge=0, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str


class RelevancyResult(BaseModel):
    score: int = Field(ge=0, le=5)
    reasoning: str


def judge_faithfulness(question: str, answer: str, context: str) -> FaithfulnessResult:
    llm = get_judge_llm().with_structured_output(FaithfulnessResult, method="json_mode")
    prompt = (
        f"{FAITHFULNESS_RUBRIC}\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
        "\n\nRespond with a JSON object with keys: score (int 0-5), "
        "unsupported_claims (list of strings), reasoning (string)."
    )
    return groq_invoke(llm, prompt)


def judge_relevancy(question: str, answer: str) -> RelevancyResult:
    llm = get_judge_llm().with_structured_output(RelevancyResult, method="json_mode")
    prompt = (
        f"{RELEVANCY_RUBRIC}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"
        "\n\nRespond with a JSON object with keys: score (int 0-5), reasoning (string)."
    )
    return groq_invoke(llm, prompt)

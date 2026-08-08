"""Hallucination guardrail: flags generation output not traceable to retrieved
context. Reuses the faithfulness judge from judge.py rather than a second
grounding check — annotates (doesn't block) since that's enough to
demonstrate the mechanism with far less code, and keeps a partially-useful
answer visible instead of hiding it entirely.
"""
from judge import judge_faithfulness

FAITHFULNESS_THRESHOLD = 3  # below this (of 5), annotate the answer as flagged


def apply_guardrail(question, answer, context, threshold=FAITHFULNESS_THRESHOLD):
    result = judge_faithfulness(question, answer, context)
    flagged = result.score < threshold

    annotated = answer
    if flagged:
        claims = "\n".join(f"  - {c}" for c in result.unsupported_claims) or "  (see reasoning)"
        annotated = (
            f"{answer}\n\n"
            f"[GUARDRAIL WARNING: faithfulness {result.score}/5 — possibly unsupported claims:\n"
            f"{claims}\n"
            f"Reason: {result.reasoning}]"
        )

    return {
        "annotated_answer": annotated,
        "flagged": flagged,
        "faithfulness": {
            "score": result.score,
            "unsupported_claims": result.unsupported_claims,
            "reasoning": result.reasoning,
        },
    }

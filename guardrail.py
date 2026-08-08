"""Hallucination guardrail: flags generation output not traceable to retrieved
context. Reuses the faithfulness judge from judge.py rather than a second
grounding check — annotates (doesn't block) since that's enough to demonstrate
the mechanism with far less code, and keeps a partially-useful answer visible.

An answer is flagged when the judge scores it below threshold OR lists any
unsupported claim — the small judge model often identifies the unsupported span
while still giving a lenient numeric score, so the claim list is the more
reliable signal.

If the judge call fails (e.g. the shared free-tier key is rate-limited), the
generated answer is still returned unflagged with the check marked unavailable —
a secondary check shouldn't discard a successful answer.
"""
from judge import judge_faithfulness

FAITHFULNESS_THRESHOLD = 3  # below this (of 5), annotate the answer as flagged


def apply_guardrail(question, answer, context, threshold=FAITHFULNESS_THRESHOLD):
    try:
        result = judge_faithfulness(question, answer, context)
    except Exception as e:
        return {
            "annotated_answer": answer,
            "flagged": False,
            "faithfulness": {
                "score": None,
                "unsupported_claims": [],
                "reasoning": f"Faithfulness check unavailable ({type(e).__name__}).",
                "available": False,
            },
        }

    flagged = result.score < threshold or bool(result.unsupported_claims)

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
            "available": True,
        },
    }

"""Runnable self-check for the non-trivial logic (no API calls, no index needed).
python test_metrics.py
"""
from eval import score_retrieval, stratified_sample


def test_score_retrieval_hit_and_rank():
    item = {"expected_doc_ids": ["pricing"]}
    chunks = [{"doc_id": "onboarding_guide"}, {"doc_id": "pricing"}, {"doc_id": "api_reference"}]
    result = score_retrieval(item, chunks)
    assert result["hit"] == 1.0
    assert result["reciprocal_rank"] == 0.5  # first match at rank 2
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1.0


def test_score_retrieval_miss():
    item = {"expected_doc_ids": ["security_and_compliance"]}
    chunks = [{"doc_id": "onboarding_guide"}, {"doc_id": "pricing"}]
    result = score_retrieval(item, chunks)
    assert result == {"hit": 0.0, "reciprocal_rank": 0.0, "precision": 0.0, "recall": 0.0}


def test_score_retrieval_multi_doc_recall():
    item = {"expected_doc_ids": ["a", "b"]}
    chunks = [{"doc_id": "a"}, {"doc_id": "c"}, {"doc_id": "b"}]
    result = score_retrieval(item, chunks)
    assert result["recall"] == 1.0
    assert result["reciprocal_rank"] == 1.0  # first match at rank 1


def test_stratified_sample_spans_scores_and_respects_n():
    rows = [{"id": f"q{i}", "faithfulness_score": i % 6} for i in range(30)]
    sample = stratified_sample(rows, 12)
    assert len(sample) == 12
    scores_seen = {r["faithfulness_score"] for r in sample}
    assert len(scores_seen) >= 5  # spans nearly the whole 0-5 range, not clustered


def test_stratified_sample_caps_at_available_rows():
    rows = [{"id": "q0", "faithfulness_score": 3}]
    sample = stratified_sample(rows, 18)
    assert len(sample) == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")

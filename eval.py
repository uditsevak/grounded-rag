"""Standalone eval harness. Run: python eval.py [--skip-calibration]

Produces eval_report.md + eval_report.json covering:
  1. Retrieval metrics (hit-rate@k, MRR, precision, recall) for dense-only vs hybrid
  2. Generation metrics (LLM-judge faithfulness + relevancy) on the hybrid path
  3. Human-vs-judge calibration on a stratified sample of faithfulness scores
"""
import argparse
import json
import statistics
from pathlib import Path

from ingest import BM25_CORPUS_PATH, build_index
from judge import judge_relevancy
from providers import GROQ_CHAT_MODEL, GROQ_JUDGE_MODEL
from rag import answer as rag_answer
from retrieval import HybridRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
REPORT_MD_PATH = Path(__file__).parent / "eval_report.md"
REPORT_JSON_PATH = Path(__file__).parent / "eval_report.json"


def load_golden_set():
    return json.loads(GOLDEN_SET_PATH.read_text())


# ---------- retrieval metrics ----------

def score_retrieval(item, retrieved_chunks):
    """retrieved_chunks: ordered list of chunk dicts (best first)."""
    expected = set(item["expected_doc_ids"])
    retrieved_doc_ids = [c["doc_id"] for c in retrieved_chunks]

    hit = 1.0 if any(d in expected for d in retrieved_doc_ids) else 0.0

    rr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            rr = 1.0 / rank
            break

    relevant_in_topk = sum(1 for d in retrieved_doc_ids if d in expected)
    precision = relevant_in_topk / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    recall = len(set(retrieved_doc_ids) & expected) / len(expected) if expected else 0.0

    return {"hit": hit, "reciprocal_rank": rr, "precision": precision, "recall": recall}


def run_retrieval_eval(retriever, golden, k, alpha):
    results = {"dense": [], "hybrid": []}
    for item in golden:
        for mode in results:
            chunks = retriever.retrieve(item["question"], k=k, mode=mode, alpha=alpha)
            results[mode].append({"id": item["id"], **score_retrieval(item, chunks)})

    summary = {}
    for mode, rows in results.items():
        summary[mode] = {
            "hit_rate@k": statistics.mean(r["hit"] for r in rows),
            "mrr": statistics.mean(r["reciprocal_rank"] for r in rows),
            "precision@k": statistics.mean(r["precision"] for r in rows),
            "recall@k": statistics.mean(r["recall"] for r in rows),
        }
    return summary, results


# ---------- generation metrics ----------

def run_generation_eval(retriever, golden, k, alpha):
    rows = []
    for item in golden:
        result = rag_answer(item["question"], retriever, mode="hybrid", k=k, alpha=alpha)
        relevancy = judge_relevancy(item["question"], result["raw_answer"])
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "raw_answer": result["raw_answer"],
                "context": result["context"],
                "sources": result["sources"],
                "faithfulness_score": result["faithfulness"]["score"],
                "faithfulness_reasoning": result["faithfulness"]["reasoning"],
                "unsupported_claims": result["faithfulness"]["unsupported_claims"],
                "relevancy_score": relevancy.score,
                "relevancy_reasoning": relevancy.reasoning,
                "flagged": result["flagged"],
            }
        )

    summary = {
        "mean_faithfulness": statistics.mean(r["faithfulness_score"] for r in rows),
        "mean_relevancy": statistics.mean(r["relevancy_score"] for r in rows),
        "flagged_count": sum(1 for r in rows if r["flagged"]),
        "flagged_rate": sum(1 for r in rows if r["flagged"]) / len(rows),
    }
    return summary, rows


# ---------- human calibration ----------

def stratified_sample(rows, n):
    """Bucket by faithfulness score (0-5), round-robin across buckets so the
    sample spans the score range rather than clustering at whatever score is
    most common."""
    buckets = {s: [] for s in range(6)}
    for r in rows:
        buckets[r["faithfulness_score"]].append(r)
    for b in buckets.values():
        b.sort(key=lambda r: r["id"])

    sample, i = [], 0
    while len(sample) < min(n, len(rows)):
        for score in range(6):
            if i < len(buckets[score]):
                sample.append(buckets[score][i])
                if len(sample) >= min(n, len(rows)):
                    break
        i += 1
    return sample


def run_calibration(generation_rows, n):
    sample = stratified_sample(generation_rows, n)

    print(f"\n--- Human calibration: rate faithfulness (0-5) for {len(sample)} items ---")
    print("(grounded in context only — same rubric the judge used, score not shown yet)\n")

    human_scores, judge_scores = [], []
    for r in sample:
        print(f"[{r['id']}] Q: {r['question']}")
        print(f"Context:\n{r['context']}\n")
        print(f"Answer: {r['raw_answer']}\n")
        while True:
            raw = input("Your faithfulness score (0-5): ").strip()
            try:
                human = int(raw)
                if 0 <= human <= 5:
                    break
            except ValueError:
                pass
            print("Enter an integer 0-5.")
        human_scores.append(human)
        judge_scores.append(r["faithfulness_score"])
        print(f"(judge scored this {r['faithfulness_score']}/5)\n{'-' * 40}\n")

    within_1 = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1) / len(sample)
    correlation = (
        statistics.correlation(human_scores, judge_scores)
        if len(set(human_scores)) > 1 and len(set(judge_scores)) > 1
        else None
    )

    return {
        "n": len(sample),
        "pct_within_1_point": within_1,
        "pearson_correlation": correlation,
        "items": [
            {"id": r["id"], "human_score": h, "judge_score": j}
            for r, h, j in zip(sample, human_scores, judge_scores)
        ],
    }


# ---------- report ----------

def write_report(config, retrieval_summary, generation_summary, calibration, generation_rows):
    report = {
        "config": config,
        "retrieval_metrics": retrieval_summary,
        "generation_metrics": generation_summary,
        "calibration": calibration,
        "generation_detail": generation_rows,
    }
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2))

    lines = ["# RAG Eval Report", ""]
    lines.append(
        f"Config: k={config['k']}, alpha={config['alpha']}, "
        f"generation_model={config['generation_model']}, judge_model={config['judge_model']}, "
        f"n_questions={config['n_questions']}"
    )
    lines.append("")

    lines.append("## Retrieval metrics (dense vs hybrid)")
    lines.append("")
    lines.append("| mode | hit-rate@k | MRR | precision@k | recall@k |")
    lines.append("|---|---|---|---|---|")
    for mode, m in retrieval_summary.items():
        lines.append(
            f"| {mode} | {m['hit_rate@k']:.3f} | {m['mrr']:.3f} | {m['precision@k']:.3f} | {m['recall@k']:.3f} |"
        )
    lines.append("")

    lines.append("## Generation metrics (hybrid path, LLM-as-judge)")
    lines.append("")
    lines.append(f"- Mean faithfulness: {generation_summary['mean_faithfulness']:.2f}/5")
    lines.append(f"- Mean relevancy: {generation_summary['mean_relevancy']:.2f}/5")
    lines.append(
        f"- Guardrail-flagged answers: {generation_summary['flagged_count']}/{config['n_questions']} "
        f"({generation_summary['flagged_rate']:.1%})"
    )
    lines.append("")

    lines.append("## Human calibration")
    lines.append("")
    if calibration is None:
        lines.append("Skipped (--skip-calibration).")
    else:
        corr = f"{calibration['pearson_correlation']:.3f}" if calibration["pearson_correlation"] is not None else "n/a (no variance)"
        lines.append(f"- Sample size: {calibration['n']}")
        lines.append(f"- % within 1 point of judge: {calibration['pct_within_1_point']:.1%}")
        lines.append(f"- Pearson correlation (human vs judge): {corr}")
        lines.append("")
        lines.append("| id | human score | judge score |")
        lines.append("|---|---|---|")
        for it in calibration["items"]:
            lines.append(f"| {it['id']} | {it['human_score']} | {it['judge_score']} |")
    lines.append("")

    REPORT_MD_PATH.write_text("\n".join(lines))
    print(f"\nWrote {REPORT_MD_PATH} and {REPORT_JSON_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5, help="dense weight in RRF fusion (0-1)")
    parser.add_argument("--calibration-n", type=int, default=18)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    if not BM25_CORPUS_PATH.exists():
        print("No index found — building it first (python ingest.py)...")
        build_index()

    golden = load_golden_set()
    retriever = HybridRetriever()

    print(f"Running retrieval eval on {len(golden)} questions (dense vs hybrid)...")
    retrieval_summary, _ = run_retrieval_eval(retriever, golden, args.k, args.alpha)

    print("Running generation eval (LLM calls, this takes a bit)...")
    generation_summary, generation_rows = run_generation_eval(retriever, golden, args.k, args.alpha)

    calibration = None
    if not args.skip_calibration:
        calibration = run_calibration(generation_rows, args.calibration_n)

    config = {
        "k": args.k,
        "alpha": args.alpha,
        "generation_model": GROQ_CHAT_MODEL,
        "judge_model": GROQ_JUDGE_MODEL,
        "n_questions": len(golden),
    }
    write_report(config, retrieval_summary, generation_summary, calibration, generation_rows)


if __name__ == "__main__":
    main()

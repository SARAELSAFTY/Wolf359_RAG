"""
Golden-set retrieval evaluation. Run after any retrieval change to catch
regressions rather than relying on manual spot-checks.

This is an INTEGRATION check: it needs a live Qdrant collection plus valid
Jina and Cloudflare credentials. It exits non-zero when quality drops below
the configured thresholds, so it can gate a deploy.

Usage:
    python run_eval.py
    python run_eval.py --min-hit-rate 0.8 --min-mrr 0.6
"""
import argparse
import math
import sys

from core import retrieval

STRESS_TEST_SET = [
    {"category": "direct", "query": "The Decima project is my life's work", "expect_character": "HILBERT"},
    {"category": "direct", "query": "Hera, I'm picking up a little interference on the transmitter", "expect_character": "MINKOWSKI"},
    {"category": "direct", "query": "Semantics, Doctor", "expect_character": "MINKOWSKI"},
    {"category": "direct", "query": "I will not dignify such an idiotic question with a response", "expect_character": "EIFFEL"},
    {"category": "paraphrased", "query": "being in multiple places at once", "expect_character": "HERA"},
    {"category": "paraphrased", "query": "a secret experiment Hilbert is running", "expect_character": "HILBERT"},
    {"category": "paraphrased", "query": "a character expressing fear of being alone forever", "expect_character": "HERA"},
    {"category": "paraphrased", "query": "someone describing an itchy spacesuit", "expect_character": "EIFFEL"},
    {"category": "paraphrased", "query": "a captain giving a direct order to her crew", "expect_character": "MINKOWSKI"},
    {"category": "filtered", "query": "door opening", "character": "EIFFEL", "expect_character": "EIFFEL"},
    {"category": "filtered", "query": "giving an order", "character": "MINKOWSKI", "expect_character": "MINKOWSKI"},
    {"category": "filtered", "query": "talking about an experiment", "character": "HILBERT", "expect_character": "HILBERT"},
    {"category": "minisode", "query": "audio log of Communications Officer Doug Eiffel installment one", "expect_character": "EIFFEL"},
    {"category": "minisode", "query": "Interim Communication Port rudimentary thought processing", "expect_character": "RACHEL"},
    {"category": "minisode", "query": "Are space suits itchy", "expect_character": "EIFFEL"},
    {"category": "minisode", "query": "diary of the second-in-command mutineers took the bridge", "expect_character": "HUI"},
    {"category": "scene", "query": "The Veldt no Eiffel", "expect_character": "HILBERT"},
]

CONFIDENCE_GAP_SET = [
    {"category": "noise_term", "query": "BEAT"},
    {"category": "noise_term", "query": "STATIC"},
    {"category": "out_of_domain", "query": "what's the weather like today"},
    {"category": "out_of_domain", "query": "how do I bake a chocolate cake"},
    {"category": "ambiguous", "query": "what happened"},
]


def evaluate_retrieval(test_set, top_k=5, use_query_expansion=True, use_scene_stitching=True, label=""):
    hits, reciprocal_ranks, context_hits, ndcgs, rows = 0, [], 0, [], []

    for case in test_set:
        if case.get("expect_character") is None:
            continue

        kwargs = {"top_k": top_k, "use_query_expansion": use_query_expansion, "use_scene_stitching": use_scene_stitching}
        if case.get("character"):
            kwargs["character"] = case["character"]

        results = retrieval.search(case["query"], **kwargs)

        rank_of_hit = None
        for rank, r in enumerate(results, start=1):
            if r["hit"].payload.get("character name") == case["expect_character"]:
                rank_of_hit = rank
                break

        found = rank_of_hit is not None
        context_found = any(
            case["expect_character"] in (r.get("scene_context") or "")
            for r in results
        )
        hits += found
        context_hits += context_found
        reciprocal_ranks.append(1 / rank_of_hit if found else 0)
        ndcgs.append(1 / math.log2(rank_of_hit + 1) if found else 0)
        rows.append({
            "query": case["query"],
            "category": case.get("category"),
            "found": found,
            "context_found": context_found,
            "rank": rank_of_hit,
        })

    scored = [c for c in test_set if c.get("expect_character") is not None]
    total = len(scored)
    hit_rate = hits / total if total else 0
    context_hit_rate = context_hits / total if total else 0
    mrr = sum(reciprocal_ranks) / total if total else 0
    ndcg = sum(ndcgs) / total if total else 0

    print(
        f"{label}  Hit Rate@{top_k}: {hit_rate:.0%}  |  MRR: {mrr:.3f}  |  "
        f"Context Hit: {context_hit_rate:.0%}  |  nDCG: {ndcg:.3f}"
    )
    for r in rows:
        status = f"rank {r['rank']}" if r["found"] else "NOT FOUND"
        context_status = "ctx-ok" if r["context_found"] else "ctx-miss"
        print(f"  {status:12} {context_status:8} {r['query']!r}")

    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "context_hit_rate": context_hit_rate,
        "ndcg": ndcg,
        "rows": rows,
    }


def evaluate_confidence_gap(test_set):
    for case in test_set:
        results = retrieval.search(case["query"], top_k=1)
        if not results:
            print(f"  {case['category']:14} {case['query']!r:55} -> EMPTY")
            continue
        top_score = results[0]["rerank_score"]
        flag = "unexpectedly confident" if top_score > 0.5 else "appropriately low confidence"
        print(f"  {case['category']:14} {case['query']!r:55} -> score={top_score:.3f}  {flag}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live retrieval evaluation (integration).")
    parser.add_argument("--min-hit-rate", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.50)
    parser.add_argument("--min-context-hit-rate", type=float, default=0.75)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("INTEGRATION EVAL - requires live Qdrant, Jina, and Cloudflare credentials.\n")

    try:
        print("=== BASELINE (no expansion, no stitching — matches original eval) ===")
        evaluate_retrieval(
            STRESS_TEST_SET,
            use_query_expansion=False,
            use_scene_stitching=False,
            label="Baseline",
        )

        print("\n=== WITH QUERY EXPANSION + SCENE STITCHING ===")
        upgraded = evaluate_retrieval(
            STRESS_TEST_SET,
            use_query_expansion=True,
            use_scene_stitching=True,
            label="Upgraded",
        )

        print("\n=== Confidence Gap ===")
        evaluate_confidence_gap(CONFIDENCE_GAP_SET)
    except Exception as exc:
        print(f"\nEVAL ERROR: {exc}", file=sys.stderr)
        return 2

    failures = []
    if upgraded["hit_rate"] < args.min_hit_rate:
        failures.append(
            f"Hit Rate {upgraded['hit_rate']:.0%} < required {args.min_hit_rate:.0%}"
        )
    if upgraded["mrr"] < args.min_mrr:
        failures.append(f"MRR {upgraded['mrr']:.3f} < required {args.min_mrr:.3f}")
    if upgraded["context_hit_rate"] < args.min_context_hit_rate:
        failures.append(
            f"Context Hit Rate {upgraded['context_hit_rate']:.0%} < required "
            f"{args.min_context_hit_rate:.0%}"
        )

    if failures:
        print("\nEVAL FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nEVAL PASSED: retrieval met the configured thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

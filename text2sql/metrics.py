"""Aggregate per-example results into the numbers reported in the README."""


def summarize(samples: list[dict]) -> dict:
    scored = [s for s in samples if s["correct"] is not None]  # drop examples whose gold SQL doesn't run
    n = len(scored)
    if not n:
        return {"n_scored": 0}
    return {
        "n_scored": n,
        "execution_accuracy": round(sum(s["correct"] for s in scored) / n, 4),
        "exact_match": round(sum(s["exact"] for s in scored) / n, 4),
        "valid_sql": round(sum(s["valid"] for s in scored) / n, 4),
        "skipped_bad_gold": len(samples) - n,
    }

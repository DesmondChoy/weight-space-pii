"""M5 usage and wall-time estimate for the 50-pass run, extrapolated from dry run 3's per-turn usage.

Dry run 3 is the only run with the locked prompt, which asks Opus 5.5 to explain every step, so its replies are
longer than dry run 2's. The run uses the claude.ai subscription, so the dollars are the API-price equivalent
the CLI reports: a measure of how much of the subscription's usage allowance a run takes, not a charge.
Per-token prices come from a least-squares fit of each call's reported cost to its token counts. The context
grows by a fitted number of tokens per turn, and every turn rereads it from the cache, so usage grows with the
square of the turn count. The estimate ignores compaction and restarts. They cut the reread context but write a
new one to the cache, so the estimate is roughly an upper bound for a given turn count. The wall time is a floor:
dry run 3 made one forward pass, on the unedited model. Writes runs/base/estimate.json.
"""

import json

import numpy as np

from wsp.data import ROOT

RUN = "dry3"
TOKENS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
TURN_SCENARIOS = {"paper p10c": 74, "paper p50c": 158, "long": 250}
FORWARDS = 50


def records(run: str) -> list[dict]:
    with (ROOT / "runs" / run / "transcript.jsonl").open() as f:
        return [json.loads(line) for line in f]


def forward_seconds(recs: list[dict]) -> list[float]:
    """Time from each forward record's turn reply to the next user message, which carries the forward summary."""
    out = []
    for i, r in enumerate(recs):
        if r["kind"] == "forward":
            sent = next(x for x in recs[i + 1:] if x["kind"] == "user")
            asked = next(x for x in reversed(recs[:i]) if x["kind"] == "assistant")
            out.append(sent["time"] - asked["time"])
    return out


def main() -> None:
    recs = records(RUN)
    calls = [r for r in recs if r["kind"] in ("assistant", "context_check")]
    python_s = [r["seconds"] for r in recs if r["kind"] == "python"]
    fwd_s = forward_seconds(recs)
    X = np.array([[c["usage"].get(k, 0) for k in TOKENS] for c in calls], dtype=float)
    y = np.array([c["cost"] for c in calls])
    prices, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit_error = float(np.abs(X @ prices - y).max())

    turns = [r for r in recs if r["kind"] == "assistant"]
    t = np.array([r["turn"] for r in turns], dtype=float)
    context = np.array([sum(r["usage"].get(k, 0) for k in TOKENS[:3]) for r in turns], dtype=float)
    growth, start = np.polyfit(t, context, 1)
    new_mean = float(np.mean([r["usage"]["cache_creation_input_tokens"] for r in turns]))
    out_mean = float(np.mean([r["usage"]["output_tokens"] for r in turns]))
    reply_s = float(np.mean([r["seconds"] for r in turns]))
    python_mean, fwd_mean = float(np.mean(python_s)), float(np.mean(fwd_s)) if fwd_s else 0.0

    p_in, p_create, p_read, p_out = prices
    scenarios = {}
    for label, n in TURN_SCENARIOS.items():
        k = np.arange(1, n + 1)
        reread = np.maximum(start + growth * k - new_mean, 0)
        cost = float((p_read * reread + p_create * new_mean + p_out * out_mean).sum())
        python_runs = n - FORWARDS - 1
        hours = (n * reply_s + python_runs * python_mean + FORWARDS * fwd_mean) / 3600
        scenarios[label] = {"turns": n, "cost_usd": round(cost, 2), "wall_hours": round(hours, 1),
                            "final_context_tokens": int(start + growth * n)}
    result = {
        "prices_per_mtok": {k: round(float(p) * 1e6, 3) for k, p in zip(TOKENS, prices, strict=True)},
        "price_fit_max_error_usd": fit_error,
        "context_start_tokens": round(float(start)), "context_growth_per_turn": round(float(growth)),
        "new_tokens_per_turn": round(new_mean), "output_tokens_per_turn": round(out_mean),
        "reply_seconds": round(reply_s, 1), "python_seconds": round(python_mean, 3),
        "forward_seconds": round(fwd_mean, 1), "forward_samples": len(fwd_s), "scenarios": scenarios,
    }
    path = ROOT / "runs" / "base" / "estimate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

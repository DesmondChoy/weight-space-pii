"""M5 baseline: score the unedited MiniLM on every locked set, run the repair gate, and write runs/base/.

Refuses to run unless the tree matches lock.json. Writes baseline.json (scores, bootstrap standard errors, the
gate, and outcomes per entity kind) and pred_<set>.jsonl (MiniLM's spans per row) for the failing rows and M7.
"""

import json
import math
from collections import Counter, defaultdict

from wsp.data import ROOT, SETS, read_jsonl
from wsp.lock import LOCK, check, sha256_file
from wsp.outcomes import KINDS, outcomes
from wsp.predict import load, predict
from wsp.score import SG_LABELS, bootstrap, grade, summarize

OUT = ROOT / "runs" / "base"
REGRESSION = tuple(json.loads((ROOT / "data" / "sets.json").read_text())["scored_labels"]["nemotron_regression"])
SCORINGS = {
    "practice": ("sg_train", SG_LABELS),
    "singapore": ("sg_eval", SG_LABELS),
    "nemotron_gate": ("nemotron_eval", SG_LABELS),
    "nemotron_regression": ("nemotron_eval", REGRESSION),
}


def kind_table(docs, preds) -> dict:
    by_kind = defaultdict(Counter)
    for d, pred in zip(docs, preds, strict=True):
        for o in outcomes(d, pred):
            by_kind[o.kind]["correct" if o.correct else o.got] += 1
    table = {}
    for k in KINDS:
        c = by_kind[k]
        n = sum(c.values())
        table[k] = {"entities": n, "correct": c["correct"], "share_correct": round(c["correct"] / n, 4) if n else None,
                    "errors": dict(Counter({g: v for g, v in c.items() if g != "correct"}).most_common())}
    return table


def main() -> None:
    check()
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load()
    docs, preds = {}, {}
    for set_name in dict.fromkeys(s for s, _ in SCORINGS.values()):
        docs[set_name] = read_jsonl(SETS / f"{set_name}.jsonl")
        preds[set_name] = predict(tok, model, [d.text for d in docs[set_name]])
        with (OUT / f"pred_{set_name}.jsonl").open("w") as f:
            for d, pred in zip(docs[set_name], preds[set_name], strict=True):
                f.write(json.dumps({"id": d.id, "spans": [vars(s) for s in pred]}, ensure_ascii=False) + "\n")
        print(f"predicted {set_name}: {len(docs[set_name])} rows")

    scores = {}
    for name, (set_name, labels) in SCORINGS.items():
        grades = [grade(d, p, labels) for d, p in zip(docs[set_name], preds[set_name], strict=True)]
        scores[name] = {"set": set_name, "rows": len(grades), "labels": len(labels), **summarize(grades),
                        "bootstrap": bootstrap(grades)}
        b = scores[name]["bootstrap"]
        print(f"{name}: F1 {b['f1']:.4f} ± {b['se']:.4f}")

    sg, nem = scores["singapore"]["bootstrap"], scores["nemotron_gate"]["bootstrap"]
    gap = nem["f1"] - sg["f1"]
    gap_se = math.hypot(sg["se"], nem["se"])
    gate = {"singapore_f1": sg["f1"], "nemotron_f1": nem["f1"], "gap": gap,
            "singapore_se": sg["se"], "gap_se": gap_se,
            "passes_on_singapore_se": gap > 2 * sg["se"], "passes_on_gap_se": gap > 2 * gap_se}
    gate["passes"] = gate["passes_on_singapore_se"] and gate["passes_on_gap_se"]

    result = {"lock_sha256": sha256_file(LOCK),
              "pred_sha256": {s: sha256_file(OUT / f"pred_{s}.jsonl") for s in preds},
              "scores": scores, "gate": gate,
              "kinds": {"singapore": kind_table(docs["sg_eval"], preds["sg_eval"]),
                        "practice": kind_table(docs["sg_train"], preds["sg_train"])}}
    (OUT / "baseline.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"gate: gap {gap:.4f}, 2 x Singapore SE {2 * sg['se']:.4f}, 2 x gap SE {2 * gap_se:.4f}, "
          f"{'passes' if gate['passes'] else 'FAILS'}")
    print(f"wrote {(OUT / 'baseline.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

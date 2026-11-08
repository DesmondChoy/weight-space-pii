"""Numbers behind the 2026-11-08 amendments to the M7 and p50b reports, from saved predictions only.

Reads data/sets/, runs/base/ and runs/<run>/eval/ for p50 and p50b. Runs no model and writes nothing. Prints:
how much of each run's Singapore F1 gain comes from each scored label, how often the targeted titles occur in each
hidden set, how many of the one-word given names MiniLM misses before and after the edit look like chat speaker
labels, the national_id tags that the Singapore scoring drops because they sit on an unscored ID span, and Nemotron
national_id precision.
"""

import json
import re
from collections import Counter

from wsp.data import ROOT, SETS, Span, read_jsonl
from wsp.score import SG_LABELS, Tally, _overlaps, merged, scored_label, split_names

RUNS = ("p50", "p50b")
TITLES = ("Madame", "Master", "Miss", "Mstr", "Mayoress", "Mister")
SPEAKER = re.compile(r"([ \t]+[^\s:(]+){0,2}([ \t]*\([^)\n]*\))?[ \t]*:")


def evaluation(run: str) -> dict:
    return json.loads((ROOT / "runs" / run / "eval" / "evaluation.json").read_text())


def tally(row: dict) -> Tally:
    return Tally(row["tp"], row["fp"], row["fn"])


def gain_split(run: str) -> dict:
    sg = evaluation(run)["scores"]["singapore"]
    base = {lab: tally(sg["base"]["by_label"][lab]) for lab in SG_LABELS}
    edited = {lab: tally(sg["edited"]["by_label"][lab]) for lab in SG_LABELS}
    f1 = lambda by: sum(by.values(), Tally()).f1
    out = {f"{lab} only": f1(base | {lab: edited[lab]}) - f1(base) for lab in SG_LABELS}
    return out | {"both": f1(edited) - f1(base)}


def title_counts() -> dict:
    texts = {name: [d.text for d in read_jsonl(SETS / f"{name}.jsonl")] for name in ("sg_eval", "nemotron_eval")}
    return {w: {name: sum(len(re.findall(rf"\b{w}\b", t)) for t in ts) for name, ts in texts.items()} for w in TITLES}


def predictions(path) -> list[list[Span]]:
    return [[Span(s["start"], s["end"], s["label"]) for s in json.loads(line)["spans"]] for line in path.open()]


def untagged_given_names(docs, base, edited) -> tuple[int, int]:
    """One-word given names that no prediction overlaps before or after the edit, and how many of them look like a
    chat speaker label: a colon follows on the same line, after at most two words and an optional bracket."""
    untagged = speakers = 0
    for d, b, e in zip(docs, base, edited, strict=True):
        for g in d.spans:
            if g.label != "first_name" or len(d.value(g).split()) > 1 or any(_overlaps(g, s) for s in b + e):
                continue
            untagged += 1
            speakers += SPEAKER.match(d.text, g.end) is not None
    return untagged, speakers


def dropped_national_id(docs, preds) -> Counter:
    """Scored national_id predictions that match no gold entity and overlap an ignore span, as score.grade drops them."""
    out = Counter()
    for d, pred in zip(docs, preds, strict=True):
        gold = {Span(s.start, s.end, merged(s.label, SG_LABELS)) for s in split_names(d.text, d.spans)}
        for s in split_names(d.text, pred):
            s = Span(s.start, s.end, scored_label(d.text, s, SG_LABELS))
            if s.label == "national_id" and s not in gold:
                if ignored := [i.label for i in d.ignore if _overlaps(s, i)]:
                    out[ignored[0]] += 1
    return out


def tags_on(docs, preds, openpii_label: str) -> Counter:
    """The label of the first prediction on each ignore span with this openpii label, or O if none overlaps it."""
    out = Counter()
    for d, pred in zip(docs, preds, strict=True):
        for i in d.ignore:
            if i.label == openpii_label:
                out[next((s.label for s in pred if _overlaps(s, i)), "O")] += 1
    return out


def main() -> None:
    print("Singapore F1 gain when only one label's tallies change")
    for run in RUNS:
        print(f"  {run}: " + ", ".join(f"{k} {v:+.4f}" for k, v in gain_split(run).items()))

    print("\nOccurrences of each targeted title, as a whole word")
    for w, n in title_counts().items():
        print(f"  {w:<9} sg_eval {n['sg_eval']:>4}   nemotron_eval {n['nemotron_eval']:>4}")

    docs = read_jsonl(SETS / "sg_eval.jsonl")
    sources = {"base": ROOT / "runs" / "base" / "pred_sg_eval.jsonl"}
    sources |= {run: ROOT / "runs" / run / "eval" / "pred_sg_eval.jsonl" for run in RUNS}
    preds = {name: predictions(path) for name, path in sources.items()}

    print("\nOne-word given names untagged before and after the edit, and those that look like chat speaker labels")
    for run in RUNS:
        untagged, speakers = untagged_given_names(docs, preds["base"], preds[run])
        print(f"  {run}: {untagged} untagged, {speakers} speaker labels")
    print("\nSingapore national_id tags dropped by the ignore rule, by the ignore span's openpii label")
    for name, pred in preds.items():
        c = dropped_national_id(docs, pred)
        print(f"  {name:<5} total {sum(c.values()):>4}  " + ", ".join(f"{k} {v}" for k, v in c.most_common()))

    for openpii_label in ("PASSPORTNUM", "DRIVERLICENSENUM"):
        print(f"\nHow MiniLM tags the main eval's {openpii_label} spans, top 4 labels")
        for name, pred in preds.items():
            c = tags_on(docs, pred, openpii_label)
            print(f"  {name:<5} of {sum(c.values())}: " + ", ".join(f"{k} {v}" for k, v in c.most_common(4)))

    print("\nNemotron regression check, national_id")
    for run in RUNS:
        reg = evaluation(run)["scores"]["nemotron_regression"]
        b, e = reg["base"]["by_label"]["national_id"], reg["edited"]["by_label"]["national_id"]
        print(f"  {run}: precision {b['precision']:.3f} -> {e['precision']:.3f}, "
              f"false positives {b['fp']} -> {e['fp']}, F1 {b['f1']:.3f} -> {e['f1']:.3f}")


if __name__ == "__main__":
    main()

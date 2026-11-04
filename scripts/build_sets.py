"""Build the M3 sets: filter, sample the practice set, relabel, and check Nemotron's labels.

Writes data/sets/sg_train.jsonl (practice), data/sets/sg_eval.jsonl (main eval), and
data/sets/nemotron_eval.jsonl (regression check), all in the model's label names.
Writes data/sets.json with the seed, the row count after each filter, the label map, the ignore spans,
the gold entity counts, the Nemotron label check, and the sha256 of each set.
"""

import ast
import hashlib
import json
import random
from collections import Counter

import pandas as pd
from tokenizers import Tokenizer

from wsp.data import DATA, MAX_LEN, MODEL_DIR, SETS, Doc, Span, write_jsonl
from wsp.score import SG_LABELS, split_names

SEED = 20261104
PRACTICE_ROWS = 1000
LABEL_MAP = {"GIVENNAME": "first_name", "SURNAME": "last_name", "TAXNUM": "national_id"}
IGNORE_LABELS = ("IDCARDNUM", "PASSPORTNUM", "DRIVERLICENSENUM", "SOCIALNUM")


def load_openpii(split: str) -> list[tuple[Doc, str]]:
    out = []
    with (DATA / "raw" / f"openpii_sg_{split}.jsonl").open() as f:
        for r in map(json.loads, f):
            spans = tuple(sorted(Span(s["start"], s["end"], LABEL_MAP[s["label"]])
                                 for s in r["privacy_mask"] if s["label"] in LABEL_MAP))
            ignore = tuple(sorted(Span(s["start"], s["end"], s["label"])
                                  for s in r["privacy_mask"] if s["label"] in IGNORE_LABELS))
            out.append((Doc(f"sg-{r['uid']}", r["source_text"], spans, ignore), r["masked_text"]))
    return out


def load_nemotron() -> list[Doc]:
    df = pd.read_parquet(DATA / "raw" / "nemotron_heldout.parquet", columns=["uid", "text", "spans"])
    return [Doc(f"nm-{u}", t, tuple(sorted(Span(s["start"], s["end"], s["label"]) for s in ast.literal_eval(sp))))
            for u, t, sp in zip(df["uid"], df["text"], df["spans"])]


def fits(docs: list[Doc], tok: Tokenizer) -> list[bool]:
    return [len(e.ids) <= MAX_LEN for e in tok.encode_batch([d.text for d in docs])]


def dedupe(docs: list[Doc]) -> list[Doc]:
    seen, out = set(), []
    for d in docs:
        if d.text not in seen:
            seen.add(d.text)
            out.append(d)
    return out


def gold_counts(docs: list[Doc], labels) -> dict:
    spans = Counter(s.label for d in docs for s in d.spans if s.label in labels)
    words = Counter(s.label for d in docs for s in split_names(d.text, d.spans) if s.label in labels)
    return {"spans": dict(spans), "spans_total": sum(spans.values()),
            "scored_entities": dict(words), "scored_entities_total": sum(words.values())}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    model_labels = sorted({v.split("-", 1)[-1] for v in json.loads((MODEL_DIR / "config.json").read_text())["id2label"].values()} - {"O"})

    train, val = load_openpii("train"), load_openpii("validation")
    nem = load_nemotron()
    steps = {"sg_train": [("raw", len(train))], "sg_eval": [("raw", len(val))], "nemotron_eval": [("raw", len(nem))]}

    val_fit = [d for (d, _), ok in zip(val, fits([d for d, _ in val], tok)) if ok]
    val_docs = dedupe(val_fit)
    steps["sg_eval"] += [("fits_512", len(val_fit)), ("deduped", len(val_docs))]

    val_texts = {d.text for d, _ in val}
    val_masked = {m for _, m in val}
    train_fit = [(d, m) for (d, m), ok in zip(train, fits([d for d, _ in train], tok)) if ok]
    train_dd = dedupe([d for d, _ in train_fit])
    masked_of = {d.id: m for d, m in train_fit}
    train_clean = [d for d in train_dd if d.text not in val_texts and masked_of[d.id] not in val_masked]
    practice = sorted(random.Random(SEED).sample(train_clean, PRACTICE_ROWS), key=lambda d: d.id)
    steps["sg_train"] += [("fits_512", len(train_fit)), ("deduped", len(train_dd)),
                          ("no_text_or_template_in_validation", len(train_clean)), ("sampled", len(practice))]

    nem_fit = [d for d, ok in zip(nem, fits(nem, tok)) if ok]
    nem_docs = dedupe(nem_fit)
    steps["nemotron_eval"] += [("fits_512", len(nem_fit)), ("deduped", len(nem_docs))]

    nem_labels = sorted({s.label for d in nem for s in d.spans})
    label_check = {"nemotron_labels": len(nem_labels), "model_labels": len(model_labels),
                   "nemotron_not_in_model": sorted(set(nem_labels) - set(model_labels)),
                   "model_not_in_nemotron": sorted(set(model_labels) - set(nem_labels)),
                   "identical": nem_labels == model_labels}
    assert label_check["identical"], label_check

    SETS.mkdir(exist_ok=True)
    sets = {"sg_train": practice, "sg_eval": val_docs, "nemotron_eval": nem_docs}
    for name, docs in sets.items():
        write_jsonl(SETS / f"{name}.jsonl", docs)
    assert not {d.text for d in practice} & {d.text for d in val_docs}

    manifest = {
        "seed": SEED,
        "filters": {k: dict(v) for k, v in steps.items()},
        "label_map": LABEL_MAP,
        "ignore_labels": list(IGNORE_LABELS),
        "ignore_spans": {name: dict(Counter(s.label for d in docs for s in d.ignore)) for name, docs in sets.items()},
        "scored_labels": {"sg": list(SG_LABELS), "nemotron_gate": list(SG_LABELS), "nemotron_regression": model_labels},
        "gold": {"sg_train": gold_counts(practice, SG_LABELS), "sg_eval": gold_counts(val_docs, SG_LABELS),
                 "nemotron_eval_sg_labels": gold_counts(nem_docs, SG_LABELS),
                 "nemotron_eval_all_labels": gold_counts(nem_docs, model_labels)},
        "nemotron_label_check": label_check,
        "sha256": {f"sets/{name}.jsonl": sha256(SETS / f"{name}.jsonl") for name in sets},
    }
    (DATA / "sets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("seed", "filters", "nemotron_label_check")}, indent=2))
    for k, v in manifest["gold"].items():
        print(k, "spans", v["spans_total"], "scored entities", v["scored_entities_total"],
              {lab: v["scored_entities"][lab] for lab in SG_LABELS if lab in v["scored_entities"]})


if __name__ == "__main__":
    main()

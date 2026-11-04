"""Profile the openpii SG splits for M2, with Nemotron's held-out split for comparison.

Reports row counts, token lengths under the model's tokenizer, span counts per label,
words per span, the share of ID values that are NRIC-shaped, phone and postcode shapes,
and text overlap between the SG splits. Writes data/m2_profile.json and prints a summary.
"""

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAX_LEN = 512

ID_LABELS = ["IDCARDNUM", "TAXNUM", "PASSPORTNUM", "SOCIALNUM", "DRIVERLICENSENUM"]
NRIC = re.compile(r"[STFGM]\d{7}[A-Z]")
SG_PHONE = re.compile(r"(\+65[\s-]?)?[3689]\d{3}[\s-]?\d{4}")
SG_POSTCODE = re.compile(r"\d{6}")
REPEATED_TAIL = re.compile(r"Singapore (.+) \1")
ADDRESS_GAP = re.compile(r"[\s#,]{0,4}")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Doc:
    text: str
    spans: tuple[Span, ...]
    masked: str = ""

    def value(self, s: Span) -> str:
        return self.text[s.start:s.end]


def load_openpii(split: str) -> list[Doc]:
    docs = []
    with (DATA / "raw" / f"openpii_sg_{split}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            spans = tuple(Span(s["start"], s["end"], s["label"]) for s in r["privacy_mask"])
            docs.append(Doc(r["source_text"], spans, r["masked_text"]))
    return docs


def load_nemotron() -> list[Doc]:
    df = pd.read_parquet(DATA / "raw" / "nemotron_heldout.parquet", columns=["text", "spans"])
    return [Doc(t, tuple(Span(s["start"], s["end"], s["label"]) for s in ast.literal_eval(sp)))
            for t, sp in zip(df["text"], df["spans"])]


def share(n: int, d: int) -> dict:
    return {"n": n, "of": d, "share": round(n / d, 4) if d else None}


def token_lengths(docs: list[Doc], tok: Tokenizer) -> np.ndarray:
    return np.array([len(e.ids) for e in tok.encode_batch([d.text for d in docs])])


def profile(docs: list[Doc], lengths: np.ndarray) -> dict:
    spans = Counter()
    rows_with = Counter()
    words = defaultdict(Counter)
    overlaps = 0
    for d in docs:
        ordered = sorted(d.spans, key=lambda s: s.start)
        overlaps += sum(a.end > b.start for a, b in zip(ordered, ordered[1:]))
        for s in d.spans:
            spans[s.label] += 1
            words[s.label][min(len(d.value(s).split()), 4)] += 1
        for label in {s.label for s in d.spans}:
            rows_with[label] += 1

    fits = lengths <= MAX_LEN
    return {
        "rows": len(docs),
        "tokens": {"mean": round(float(lengths.mean()), 1),
                   **{f"p{q}": int(np.percentile(lengths, q)) for q in (50, 90, 99)},
                   "max": int(lengths.max()), f"fits_{MAX_LEN}": share(int(fits.sum()), len(docs))},
        "overlapping_span_pairs": overlaps,
        "labels": {label: {"spans": n, "rows": rows_with[label],
                           "words": {("4+" if k == 4 else str(k)): v for k, v in sorted(words[label].items())}}
                   for label, n in spans.most_common()},
    }


def address_order(d: Doc, ordered: list[Span], i: int) -> tuple[str, str]:
    """Where the STREET sits relative to the BUILDINGNUM at ordered[i], and the text between them."""
    for side, j in (("street_first", i - 1), ("number_first", i + 1)):
        if 0 <= j < len(ordered) and ordered[j].label == "STREET":
            a, b = sorted((ordered[i], ordered[j]), key=lambda s: s.start)
            gap = d.text[a.end:b.start]
            if ADDRESS_GAP.fullmatch(gap):
                return side, gap
    return "not_adjacent", ""


def openpii_shapes(docs: list[Doc]) -> dict:
    ids = {label: [0, 0] for label in ID_LABELS}
    phone = Counter()
    postcode = [0, 0]
    city = Counter()
    order = Counter()
    gaps = Counter()
    for d in docs:
        ordered = sorted(d.spans, key=lambda s: s.start)
        for i, s in enumerate(ordered):
            v = d.value(s).strip()
            if s.label in ids:
                ids[s.label][0] += bool(NRIC.fullmatch(v))
                ids[s.label][1] += 1
            elif s.label == "TELEPHONENUM":
                phone["total"] += 1
                phone["sg_shaped"] += bool(SG_PHONE.fullmatch(v))
                phone["sg_prefix"] += v.startswith(("+65", "0065", "65"))
            elif s.label == "ZIPCODE":
                postcode[0] += bool(SG_POSTCODE.fullmatch(v))
                postcode[1] += 1
            elif s.label == "CITY":
                city["plain_singapore" if v == "Singapore" else
                     "repeated_tail" if REPEATED_TAIL.fullmatch(v) else "other"] += 1
            elif s.label == "BUILDINGNUM":
                side, gap = address_order(d, ordered, i)
                order[side] += 1
                if side != "not_adjacent":
                    gaps[gap] += 1
    nric_total = sum(n for n, _ in ids.values())
    n_city = sum(city.values())
    n_num = sum(order.values())
    return {
        "nric_shaped": {"by_label": {k: share(*v) for k, v in ids.items()},
                        "all_id_labels": share(nric_total, sum(t for _, t in ids.values()))},
        "phone": {"sg_shaped": share(phone["sg_shaped"], phone["total"]),
                  "sg_prefix": share(phone["sg_prefix"], phone["total"])},
        "six_digit_postcode": share(*postcode),
        "city": {k: share(v, n_city) for k, v in city.most_common()},
        "buildingnum_vs_street": {k: share(v, n_num) for k, v in order.most_common()},
        "address_gaps": dict(gaps.most_common(8)),
    }


def examples(docs: list[Doc], label: str, k: int = 8) -> list[str]:
    return [d.value(s) for d in docs for s in d.spans if s.label == label][:k]


def split_overlap(a: list[Doc], b: list[Doc]) -> dict:
    return {"same_text": len({d.text for d in a} & {d.text for d in b}),
            "same_masked_text": len({d.masked for d in a} & {d.masked for d in b}),
            "dup_text_within": {"train": len(a) - len({d.text for d in a}),
                                "validation": len(b) - len({d.text for d in b})}}


def main() -> None:
    tok = Tokenizer.from_file(str(DATA / "model" / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    sets = {"sg_train": load_openpii("train"), "sg_validation": load_openpii("validation"),
            "nemotron_heldout": load_nemotron()}
    out = {name: profile(docs, token_lengths(docs, tok)) for name, docs in sets.items()}
    for name in ("sg_train", "sg_validation"):
        out[name]["shapes"] = openpii_shapes(sets[name])
        out[name]["examples"] = {label: examples(sets[name], label)
                                 for label in ["GIVENNAME", "SURNAME", "TELEPHONENUM", "IDCARDNUM", "TAXNUM", "CITY"]}
    out["sg_split_overlap"] = split_overlap(sets["sg_train"], sets["sg_validation"])
    (DATA / "m2_profile.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

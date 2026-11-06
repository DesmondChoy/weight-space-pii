"""The fixed scoring protocol: BIO repair, word-level entities, name splitting, and strict micro F1.

Word tags come from the argmax label of each word's first subtoken. Decoding turns word tags into
entities. Scoring splits every first_name and last_name entity into one entity per
whitespace-separated word, on both the gold and the predicted side. Where `name` is scored, first_name and
last_name both count as `name`. A predicted tax_id on an NRIC-shaped value counts as national_id, and scored
predictions inside a document's ignore spans are dropped. An entity is then correct only if its span and its
type both match exactly.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from wsp.data import Doc, Span

NAME = "name"
NAME_LABELS = frozenset({"first_name", "last_name"})
SG_LABELS = (NAME, "national_id")
FAILURES = ("missed", "wrong_type", "wrong_boundary", "false_positive")
NRIC = re.compile(r"[STFGM]\d{7}[A-Z]")
WORD = re.compile(r"\S+")


def decode(words: list[tuple[int, int]], tags: list[str]) -> list[Span]:
    """Turn word tags into entities. An I-X tag continues an entity only after B-X or I-X; otherwise it starts one."""
    ents: list[Span] = []
    prev = "O"
    for (start, end), tag in zip(words, tags, strict=True):
        bio, _, typ = tag.partition("-")
        if bio == "I" and prev in (f"B-{typ}", f"I-{typ}"):
            ents[-1] = Span(ents[-1].start, end, typ)
        elif bio in ("B", "I"):
            ents.append(Span(start, end, typ))
        prev = tag
    return ents


def split_names(text: str, spans: tuple[Span, ...] | list[Span]) -> list[Span]:
    out = []
    for s in spans:
        if s.label in NAME_LABELS:
            out += [Span(s.start + m.start(), s.start + m.end(), s.label) for m in WORD.finditer(text[s.start:s.end])]
        else:
            out.append(s)
    return out


@dataclass(frozen=True)
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, o: "Tally") -> "Tally":
        return Tally(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn) if self.tp + self.fp + self.fn else 0.0


@dataclass(frozen=True)
class Grade:
    """One document's result: a tally per scored label and a count per failure kind."""
    by_label: dict[str, Tally]
    failures: Counter = field(default_factory=Counter)

    @property
    def total(self) -> Tally:
        return sum(self.by_label.values(), Tally())


def merged(label: str, labels: tuple[str, ...]) -> str:
    """first_name and last_name count as one label, `name`, wherever `name` is scored.

    Which word of a name openpii calls the surname is a convention a reader often cannot check, and hiding a
    name does not depend on it. The all-label regression check scores both labels, so there it stays strict.
    """
    return NAME if label in NAME_LABELS and NAME in labels else label


def scored_label(text: str, pred: Span, labels: tuple[str, ...]) -> str:
    """The label a prediction is scored as. A tax_id on an NRIC-shaped value is scored as national_id.

    openpii puts NRICs in tax context, so tax_id is a fair type for them. The rule is off when tax_id is
    itself scored, so the all-label regression check stays strict.
    """
    if (pred.label == "tax_id" and "tax_id" not in labels and "national_id" in labels
            and NRIC.fullmatch(text[pred.start:pred.end])):
        return "national_id"
    return merged(pred.label, labels)


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def grade(doc: Doc, pred: list[Span], labels: tuple[str, ...]) -> Grade:
    """Score one document over `labels`.

    Each prediction is first relabelled by `scored_label`.
    A scored prediction that overlaps one of the document's `ignore` spans and matches no gold entity
    is dropped before counting, so it is neither right nor wrong.

    Each unmatched gold entity is one failure: wrong_type if some prediction has its exact span,
    else wrong_boundary if some prediction overlaps it, else missed. A prediction with a scored label
    that overlaps no gold entity with a scored label is a false_positive. Predictions with other labels
    never count as false positives, but they can make a gold entity wrong_type or wrong_boundary.
    """
    gold_all = [Span(s.start, s.end, merged(s.label, labels)) for s in split_names(doc.text, doc.spans)]
    g = {s for s in gold_all if s.label in labels}
    pred_all = [Span(s.start, s.end, scored_label(doc.text, s, labels)) for s in split_names(doc.text, pred)]
    p = {s for s in pred_all if s.label in labels and (s in g or not any(_overlaps(s, i) for i in doc.ignore))}
    pred_by_span = {(s.start, s.end) for s in pred_all}
    failures = Counter()
    for s in g - p:
        if (s.start, s.end) in pred_by_span:
            failures["wrong_type"] += 1
        elif any(_overlaps(q, s) for q in pred_all):
            failures["wrong_boundary"] += 1
        else:
            failures["missed"] += 1
    failures["false_positive"] = sum(not any(_overlaps(q, s) for s in g) for q in p - g)
    by_label = {lab: Tally(len({s for s in g & p if s.label == lab}),
                           len({s for s in p - g if s.label == lab}),
                           len({s for s in g - p if s.label == lab})) for lab in labels}
    return Grade(by_label, +failures)


def summarize(grades: list[Grade]) -> dict:
    labels = grades[0].by_label.keys() if grades else ()
    by_label = {lab: sum((gr.by_label[lab] for gr in grades), Tally()) for lab in labels}
    total = sum(by_label.values(), Tally())
    failures = sum((gr.failures for gr in grades), Counter())

    def row(t: Tally) -> dict:
        return {"f1": round(t.f1, 4), "precision": round(t.precision, 4), "recall": round(t.recall, 4),
                "tp": t.tp, "fp": t.fp, "fn": t.fn}

    return {"overall": row(total), "by_label": {lab: row(t) for lab, t in by_label.items()},
            "failures": {k: failures[k] for k in FAILURES}}


def _counts(grades: list[Grade]) -> np.ndarray:
    return np.array([[gr.total.tp, gr.total.fp, gr.total.fn] for gr in grades], dtype=np.int64)


def _f1(c: np.ndarray) -> np.ndarray:
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def bootstrap(grades_a: list[Grade], grades_b: list[Grade] | None = None, n: int = 2000, seed: int = 0) -> dict:
    """Micro F1 with its bootstrap standard error over documents.

    With `grades_b`, also the paired difference b minus a, resampling the same documents for both.
    """
    a = _counts(grades_a)
    b = a if grades_b is None else _counts(grades_b)
    assert len(b) == len(a), "paired bootstrap needs the same documents"
    rng = np.random.default_rng(seed)
    sums_a, sums_b = [], []
    for k in range(0, n, 100):
        idx = rng.integers(0, len(a), size=(min(100, n - k), len(a)))
        sums_a.append(a[idx].sum(axis=1))
        sums_b.append(b[idx].sum(axis=1))
    fa, fb = _f1(np.concatenate(sums_a)), _f1(np.concatenate(sums_b))
    out = {"f1": float(_f1(a.sum(axis=0))), "se": float(fa.std(ddof=1))}
    if grades_b is not None:
        out["diff"] = float(_f1(b.sum(axis=0)) - out["f1"])
        out["diff_se"] = float((fb - fa).std(ddof=1))
    return out

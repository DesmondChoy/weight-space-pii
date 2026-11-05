"""What happened to each scored gold entity, grouped by the kinds the M5 prediction and the M7 comparison use."""

import re
from dataclasses import dataclass

from wsp.data import Doc, Span
from wsp.score import NAME_LABELS, SG_LABELS, scored_label, split_names

TAX_BEFORE = re.compile("tax", re.IGNORECASE)
CONTEXT_CHARS = 60
KINDS = ("first_name/one word", "first_name/multi-word, first word", "first_name/multi-word, later word",
         "last_name/one word", "last_name/multi-word, first word", "last_name/multi-word, later word",
         "national_id/tax context", "national_id/no tax context")


@dataclass(frozen=True)
class Outcome:
    """One scored gold entity. `got` is the scored label on its exact span, `boundary: <label>`, or `missed`."""
    gold: Span
    word: Span
    kind: str
    got: str

    @property
    def correct(self) -> bool:
        return self.got == self.word.label


def kind(d: Doc, gold: Span, word: Span) -> str:
    if gold.label not in NAME_LABELS:
        before = d.text[max(0, gold.start - CONTEXT_CHARS):gold.start]
        return f"{gold.label}/{'tax' if TAX_BEFORE.search(before) else 'no tax'} context"
    if len(d.value(gold).split()) == 1:
        return f"{gold.label}/one word"
    return f"{gold.label}/multi-word, {'first' if word.start == gold.start else 'later'} word"


def outcomes(d: Doc, pred: list[Span], labels: tuple[str, ...] = SG_LABELS) -> list[Outcome]:
    words = [Span(p.start, p.end, scored_label(d.text, p, labels)) for p in split_names(d.text, pred)]
    exact = {(p.start, p.end): p.label for p in words}
    out = []
    for g in d.spans:
        if g.label not in labels:
            continue
        for w in split_names(d.text, [g]):
            if (w.start, w.end) in exact:
                got = exact[(w.start, w.end)]
            else:
                over = [p for p in words if p.start < w.end and w.start < p.end]
                got = f"boundary: {over[0].label}" if over else "missed"
            out.append(Outcome(g, w, kind(d, g, w), got))
    return out

"""Spans, documents, and the JSONL format shared by the practice and evaluation sets."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL_DIR = DATA / "model"
SETS = DATA / "sets"
MAX_LEN = 512


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Doc:
    """A text with its gold spans. Scored predictions inside an `ignore` span count neither way."""
    id: str
    text: str
    spans: tuple[Span, ...]
    ignore: tuple[Span, ...] = ()

    def value(self, s: Span) -> str:
        return self.text[s.start:s.end]


def write_jsonl(path: Path, docs: list[Doc]) -> None:
    with path.open("w") as f:
        for d in docs:
            row = {"id": d.id, "text": d.text, "spans": _dump(d.spans), "ignore": _dump(d.ignore)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Doc]:
    with path.open() as f:
        return [Doc(r["id"], r["text"], _load(r["spans"]), _load(r["ignore"])) for r in map(json.loads, f)]


def _dump(spans: tuple[Span, ...]) -> list[dict]:
    return [{"start": s.start, "end": s.end, "label": s.label} for s in spans]


def _load(rows: list[dict]) -> tuple[Span, ...]:
    return tuple(Span(s["start"], s["end"], s["label"]) for s in rows)

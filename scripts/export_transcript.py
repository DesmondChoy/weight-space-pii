"""Render a run's transcript.jsonl as Markdown: every message to and from the editor, in order.

Python and forward events repeat what the messages already carry, so only messages are rendered.
"""

import argparse
import json
from pathlib import Path

HEADINGS = {"user": "Harness", "assistant": "Editor", "context_check": "Editor (context check)"}


def render(transcript: Path) -> str:
    events = [json.loads(line) for line in transcript.open()]
    out = [f"# Transcript of `{transcript.parent.name}`", ""]
    for e in events:
        heading = HEADINGS.get(e["kind"])
        if heading is None:
            continue
        cost = f", ${e['cost']:.3f}" if "cost" in e else ""
        out += [f"## Turn {e['turn']}: {heading}{cost}", "", "`````text", e["text"].rstrip(), "`````", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="run folder, for example runs/dry2")
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.write_text(render(a.run / "transcript.jsonl"))


if __name__ == "__main__":
    main()

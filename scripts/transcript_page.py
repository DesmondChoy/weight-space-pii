"""Build a browsable HTML logbook of one editor run: every message the harness sent Opus 5.5 and every reply.

Reads runs/<run>/transcript.jsonl, work_notes/notes.md, work_notes/split.json and final_answer.md. Writes one
self-contained HTML page. The page renders Markdown and highlights code with libraries from cdnjs.

Usage: uv run python -I scripts/transcript_page.py runs/p50 <out.html>

The phases and key moments are read by hand from each run, so RUNS holds them for every run the page supports.
"""

import argparse
import json
import re
from pathlib import Path

RUNS = {
    "p50": {  # From docs/m6_report.md, "How Opus 5.5 spent its 50 passes". Turn 121 is a key moment inside phase 3.
        "phases": [
            {"id": 1, "first": 1, "last": 42, "short": "Read and measure",
             "name": "Read the data and the unedited MiniLM"},
            {"id": 2, "first": 43, "last": 120, "short": "Build repairs",
             "name": "Build the repairs on the dev half"},
            {"id": 3, "first": 121, "last": 207, "short": "Test and take apart",
             "name": "Test on the held-back half, check for damage, take the edit apart"},
            {"id": 4, "first": 208, "last": 232, "short": "Confirm",
             "name": "Confirm the final edit and answer"},
        ],
        "moments": [  # (forward pass, label). The turn is looked up from the transcript, never typed in.
            (1, "First forward pass: the unedited MiniLM on the dev half"),
            (4, "First edit tested: the NRIC repair in layer 4"),
            (15, "First test on the held-back half"),
            (23, "Each repair scored alone on the held-back half (passes 23 to 25)"),
            (33, "Test that MiniLM uses the title signal"),
            (40, "Final edit first tested on the held-back half"),
            (50, "Last forward pass"),
        ],
    },
    "p50b": {  # Read from the p50b transcript: each phase ends with the last forward pass of its stage.
        "phases": [
            {"id": 1, "first": 1, "last": 35, "short": "Read and measure",
             "name": "Read the data and the unedited MiniLM"},
            {"id": 2, "first": 36, "last": 135, "short": "Build repairs",
             "name": "Build both repairs on the dev set"},
            {"id": 3, "first": 136, "last": 216, "short": "Test and tighten",
             "name": "Test on the held-back set, tighten the gates and check for damage"},
            {"id": 4, "first": 217, "last": 274, "short": "Extend and confirm",
             "name": "Add the Master gate, map Miss and Mayoress to Madam, and confirm"},
        ],
        "moments": [
            (1, "First forward pass: the unedited MiniLM on the dev set"),
            (3, "First edit: ablating seven title neurons, which changes nothing"),
            (5, "First repair tested: the NRIC gate in layer 4"),
            (6, "Both repairs together for the first time"),
            (13, "First test on the held-back set"),
            (23, "Dose-response: both repairs at 0.5, 0.75 and 1.25 strength (passes 23 to 25)"),
            (29, "Gates moved to the quietest neurons, checked on 440 US sentences"),
            (38, "Each repair scored alone on the held-back set (passes 38 and 39)"),
            (44, "Final edit first tested: Miss and Mayoress now map to Madam"),
            (45, "Final edit on the held-back set"),
            (50, "Last forward pass"),
        ],
    },
}
FENCE = re.compile(r"^```(python|forward)\s*$", re.M)


def opening(text: str, limit: int = 300) -> str:
    """The first two sentences of an explanation, cut at a word boundary."""
    text = " ".join(text.split())
    ends = [m.start() for m in re.finditer(r"(?<=[.!?])\s", text)]
    s = text[: ends[1]] if len(ends) > 1 else text
    return s if len(s) <= limit else s[: limit].rsplit(" ", 1)[0] + "…"


def compact(summary: dict) -> dict:
    out = {}
    for model in ("original", "edited"):
        if model in summary:
            s = summary[model]
            out[model] = {"f1": s["overall"]["f1"],
                          "labels": {k: v["f1"] for k, v in s["by_label"].items()}}
    if "diff" in summary:
        d = summary["diff"]
        out["diff"] = [d["f1_edited_minus_original"], d["se"], d["items_better"], d["items_worse"]]
    return out


def build(run: Path) -> dict:
    phases, moments = RUNS[run.name]["phases"], RUNS[run.name]["moments"]
    events = [json.loads(line) for line in (run / "transcript.jsonl").open()]
    split = json.loads((run / "work_notes" / "split.json").read_text())
    dev, hold = set(split["dev"]), set(split["hold"])
    users = {e["turn"]: e for e in events if e["kind"] == "user"}
    replies = {e["turn"]: e for e in events if e["kind"] == "assistant"}
    blocks = {e["turn"]: e for e in events if e["kind"] in ("python", "forward", "invalid")}
    start = users[1]["time"]
    turns, passes = [], []
    for t in sorted(replies):
        r, u, b = replies[t], users[t], blocks.get(t)
        m = FENCE.search(r["text"])
        expl = r["text"][: m.start()].strip() if m and b and b["kind"] != "invalid" else r["text"].strip()
        usage = r["usage"]
        row = {"t": t, "elapsed": round(u["time"] - start), "sent": u["text"], "expl": expl,
               "secs": round(r["seconds"], 1), "cost": r["cost"],
               "ctx": usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"],
               "out": usage["output_tokens"],
               "phase": next(p["id"] for p in phases if p["first"] <= t <= p["last"])}
        if b is None:
            row["block"] = {"kind": "final"}
        elif b["kind"] == "python":
            row["block"] = {"kind": "python", "n": b["n"], "code": b["code"], "secs": round(b["seconds"], 1),
                            "restarted": b["restarted"]}
        elif b["kind"] == "invalid":
            row["block"] = {"kind": "invalid", "blocks": b["blocks"]}
        else:
            req = json.loads(b["request"])
            items = req.get("items", [])
            half = {"dev": sum(i in dev for i in items), "held": sum(i in hold for i in items)}
            assert half["dev"] + half["held"] == len(items), f"pass {b['k']} has items outside the split"
            fwd = {"kind": "forward", "k": b["k"], "model": req["model"], "items": half,
                   "texts": len(req.get("texts", [])), "returns": [k for k, v in req.get("return", {}).items() if v],
                   "request": b["request"], "result": compact(b["summary"]), "secs": b["summary"]["seconds"]}
            row["block"] = fwd
            passes.append({"k": b["k"], "t": t, "model": req["model"], "items": half, "texts": fwd["texts"],
                           "returns": fwd["returns"], "result": fwd["result"], "why": opening(expl)})
        turns.append(row)
    for row in turns:
        prev = next((x for x in turns if x["t"] == row["t"] - 1), None)
        if prev is None:
            row["from"] = "The task prompt"
        elif prev["block"]["kind"] == "python":
            row["from"] = f"Output of Python run {prev['block']['n']}, from turn {prev['t']}"
        elif prev["block"]["kind"] == "forward":
            row["from"] = f"Result of forward pass {prev['block']['k']}, from turn {prev['t']}"
        else:
            row["from"] = f"Refusal of turn {prev['t']}, which held {prev['block']['blocks']} code blocks"
    by_pass = {p["k"]: p["t"] for p in passes}
    run_json = json.loads((run / "run.json").read_text())
    return {"run": run.name, "turns": turns, "passes": passes, "phases": phases,
            "moments": [{"k": k, "t": by_pass[k], "label": label} for k, label in moments],
            "notes": (run / "work_notes" / "notes.md").read_text(), "final": (run / "final_answer.md").read_text(),
            "stats": {"turns": run_json["turns"], "python": run_json["python_runs"], "passes": run_json["passes_used"],
                      "invalid": run_json["invalid_replies"], "wall": run_json["wall_seconds"],
                      "cost": run_json["cost_usd"], "model": run_json["model"], "effort": run_json["effort"],
                      "dev": len(dev), "held": len(hold)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    log = build(a.run)
    data = json.dumps(log, ensure_ascii=False).replace("</", "<\\/")
    page = (Path(__file__).with_name("transcript_page.html").read_text()
            .replace("__RUN__", a.run.name).replace("__LAST__", str(log["turns"][-1]["t"]))
            .replace("__DATA__", data))
    a.out.write_text(page)
    print(f"wrote {a.out} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()

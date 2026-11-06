"""Generate the figures in the M2 to M7 reports: data_sets, example_rows, label_translation, tax_id_cases,
harness, edit_checks, lock_checks, set_roles, baseline_f1, baseline_kinds, run_timeline, edit_map, m7_kinds and
m7_errors.

The M6 numbers come from runs/p50/, which git does not track: each forward pass's turn and items from
transcript.jsonl, forward/<k>/request.json and work_notes/split.json, the context from cli_session.jsonl, and
the edited neurons from edit/edit.npz. The M7 numbers come from runs/p50/eval/evaluation.json, which
scripts/evaluate.py writes."""

import json
from pathlib import Path

HERE = Path(__file__).parent
CW = 7.8
STYLE = """<style>
text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.th { font-size: 14px; font-weight: 500; fill: #2C2C2A; }
.ts { font-size: 12px; fill: #5F5E5A; }
.tl { font-size: 11px; fill: #5F5E5A; }
.mono { font-family: Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 13px; fill: #2C2C2A; }
.box rect { fill: #F1EFE8; stroke: #888780; }
.arr { stroke: #5F5E5A; stroke-width: 1.5; fill: none; }
.c-teal rect, rect.c-teal { fill: #E1F5EE; stroke: #0F6E56; } .c-teal .th { fill: #085041; } .c-teal .ts { fill: #0F6E56; }
.c-coral rect, rect.c-coral { fill: #FAECE7; stroke: #993C1D; } .c-coral .th { fill: #712B13; } .c-coral .ts { fill: #993C1D; }
rect.sc { fill: #E1F5EE; stroke: #0F6E56; } text.sc { fill: #085041; font-weight: 500; }
rect.dr { fill: none; stroke: #888780; stroke-dasharray: 4 3; }
.c-purple rect { fill: #EEEDFE; stroke: #534AB7; } .c-purple .th { fill: #3C3489; } .c-purple .ts { fill: #534AB7; }
.card { fill: #FAFAF7; stroke: #E5E3DA; }
rect.tag { fill: #FBEAF0; stroke: #993556; }
.tv { font-size: 12px; font-weight: 500; fill: #2C2C2A; }
.ok { font-size: 12px; font-weight: 500; fill: #3B6D11; } .bad { font-size: 12px; font-weight: 500; fill: #A32D2D; }
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#5F5E5A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
"""


def text(out, x, y, s, cls, anchor="middle"):
    out.append(f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


def box(out, x, y, w, h, title, lines=(), cls="box"):
    out.append(f'<g class="{cls}"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="8" stroke-width="0.5"/>')
    top = y + h / 2 - 8 * len(lines)
    text(out, x + w / 2, top, title, "th")
    for i, ln in enumerate(lines):
        s, c = ln if isinstance(ln, tuple) else (ln, "ts")
        text(out, x + w / 2, top + 18 + 16 * i, s, c)
    out.append("</g>")


def write(name, w, h, title, desc, out):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">\n'
            f"<title>{title}</title>\n<desc>{desc}</desc>\n{STYLE}"
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>\n')
    (HERE / name).write_text(head + "\n".join(out) + "\n</svg>\n")


def data_sets():
    out = []
    for x, s in ((160, "Source"), (479, "Filter"), (800, "Set")):
        text(out, x, 24, s, "ts")
    rows = [
        ("openpii train", "20,225 Singapore rows", "filter to 20,211, then sample 1,000",
         "drops long, duplicate, and shared texts", "Practice set", ["1,000 rows", "Opus 5.5 sees rows and answers"], "c-teal"),
        ("openpii validation", "5,005 Singapore rows", "filter to 5,004", "drops 1 duplicate text",
         "Main eval", ["5,004 rows", "hidden, Singapore score"], "c-coral"),
        ("Nemotron held-out", "20,000 rows, never trained on", "filter to 18,515", "drops 1,485 rows over 512 tokens",
         "Regression check", ["18,515 rows", "hidden, Nemotron score"], "c-coral"),
    ]
    for i, (src, src_sub, above, below, dst, dst_sub, cls) in enumerate(rows):
        y = 44 + 90 * i
        box(out, 40, y, 240, 66, src, [src_sub])
        out.append(f'<path d="M282 {y + 33} L676 {y + 33}" class="arr" marker-end="url(#arrow)"/>')
        text(out, 479, y + 21, above, "ts")
        text(out, 479, y + 46, below, "tl")
        box(out, 680, y, 240, 66, dst, dst_sub, cls)
    out.append('<line x1="660" y1="122" x2="940" y2="122" stroke="#888780" stroke-width="1" stroke-dasharray="4 4"/>')
    ly = 330
    for x, cls, label in ((40, "c-teal", "Opus 5.5 can see"), (200, "c-coral", "Hidden from Opus 5.5")):
        out.append(f'<rect class="{cls}" x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/>')
        text(out, x + 22, ly, label, "ts", "start")
    write("data_sets.svg", 960, 352, "Where each set comes from",
          "Three sources pass a filter into three sets. openpii train becomes the 1,000-row practice set, the only set "
          "Opus 5.5 sees. openpii validation becomes the 5,004-row main eval, and Nemotron's held-out split becomes "
          "the 18,515-row regression check. Both are hidden from Opus 5.5.", out)


def example_rows():
    out = []
    x0 = 40

    def line(y, s, marks):
        ends = [-1e9, -1e9]
        for word, cls, label in marks:
            i = s.index(word)
            x, w = x0 + i * CW - 2, len(word) * CW + 4
            out.append(f'<rect class="{cls}" x="{x:g}" y="{y - 11:g}" width="{w:g}" height="22" rx="4" stroke-width="1"/>')
            lw = len(label) * 6.2
            lx = max(x0, x + w / 2 - lw / 2)
            r = 0 if lx >= ends[0] + 8 else 1
            ends[r] = lx + lw
            text(out, lx, y + 24 + 14 * r, label, f"tl {cls}", "start")
        out.append(f'<text class="mono" x="{x0}" y="{y}" dominant-baseline="central" '
                   f'textLength="{len(s) * CW:g}" lengthAdjust="spacingAndGlyphs" xml:space="preserve">{s}</text>')

    def sc(word, src, dst):
        return word, "sc", f"{src} → {dst}"

    def dr(word, src):
        return word, "dr", src

    text(out, x0, 22, "Row sg-24648387, shortened", "th", "start")
    line(60, "Mister Enie Idilia Cibien, … scheduled for December 25th, 1944 at the Singapore Drop Zone.",
         [dr("Mister", "TITLE"), sc("Enie Idilia", "GIVENNAME", "first_name"), sc("Cibien", "SURNAME", "last_name"),
          dr("December 25th, 1944", "DATE"), dr("Singapore", "CITY")])
    line(124, "… located at 11 Lorong K Telok Kurau, … Please bring a copy of your OH1409941 or 598QI5X2NW …",
         [dr("11", "BUILDINGNUM"), dr("Lorong K Telok Kurau", "STREET"), dr("OH1409941", "PASSPORTNUM"),
          dr("598QI5X2NW", "IDCARDNUM")])
    text(out, x0, 196, "Row sg-24831906", "th", "start")
    line(234, "Our compliance officer, Deyancho Amr Kayoumi, will review",
         [sc("Deyancho Amr", "GIVENNAME", "first_name"), sc("Kayoumi", "SURNAME", "last_name")])
    line(298, "the tax details under G6346291N during the quarterly audit.",
         [sc("G6346291N", "TAXNUM", "national_id")])
    ly = 362
    out.append(f'<rect class="sc" x="{x0}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="1"/>')
    text(out, x0 + 22, ly, "Scored. openpii's label → MiniLM's label", "ts", "start")
    out.append(f'<rect class="dr" x="{x0 + 330}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="1"/>')
    text(out, x0 + 352, ly, "Not scored. openpii's label", "ts", "start")
    write("example_rows.svg", 960, 384, "Two main-eval rows and their gold spans",
          "Two openpii rows with every gold span marked. Given names, surnames, and the NRIC are scored, "
          "renamed to MiniLM's labels first_name, last_name, and national_id. Titles, dates, cities, addresses, and the passport "
          "and ID card numbers are not scored.", out)


def label_translation():
    out = []
    for x, s in ((140, "openpii gold spans"), (440, "Gold spans in MiniLM's label names"), (780, "MiniLM's output")):
        text(out, x, 24, s, "ts")
    for i, (src, sub, dst) in enumerate([("GIVENNAME", "given names", "first_name"), ("SURNAME", "surnames", "last_name"),
                                         ("TAXNUM", "every value an NRIC", "national_id")]):
        y = 44 + 72 * i
        box(out, 40, y, 200, 56, src, [sub])
        out.append(f'<path d="M242 {y + 28} L336 {y + 28}" class="arr" marker-end="url(#arrow)"/>')
        box(out, 340, y, 200, 56, dst, cls="c-purple")
    box(out, 640, 44, 280, 200, "MiniLM's tags",
        ["55 types, such as first_name,", "national_id, and tax_id", "", "The types never change.",
         "Only the weights are edited."])
    out.append('<path d="M440 246 L440 280" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M780 246 L780 280" class="arr" marker-end="url(#arrow)"/>')
    box(out, 340, 282, 580, 56, "Scoring code", ["A tag counts only if its words and its type match the gold spans."], "c-purple")
    write("label_translation.svg", 960, 360, "The gold spans are translated, not MiniLM",
          "The gold spans are the correct answers marked in the data. openpii's labels GIVENNAME, SURNAME, and "
          "TAXNUM are renamed to MiniLM's labels first_name, last_name, and national_id. MiniLM's 55 tag types never "
          "change. The scoring code compares MiniLM's tags with the renamed gold spans.", out)


def tax_id_cases():
    out = []
    text(out, 40, 22, "How the scoring code treats a tax_id tag", "th", "start")
    text(out, 40, 42, "The scoring code reads a tax_id tag as national_id only when the tagged text is NRIC-shaped, "
         "a letter, seven digits, and a letter.", "ts", "start")
    cases = [
        ("A real NRIC", "the tax details under G6346291N", "G6346291N",
         [("MiniLM's tag", "tax_id"), ("Gold span", "national_id"), ("Scored as", "national_id")],
         "correct", "ok"),
        ("An NRIC with no gold span", "IDs S1234567D and T7654321A.", "T7654321A",
         [("MiniLM's tag", "tax_id"), ("Gold span", "nothing here"), ("Scored as", "national_id")],
         "mistake, a false positive", "bad"),
        ("A US ID number in Nemotron", "SSN 123-45-6789.", "123-45-6789",
         [("MiniLM's tag", "tax_id"), ("Gold span", "national_id"), ("Scored as", "tax_id, not NRIC-shaped")],
         "mistake, wrong type", "bad"),
    ]
    for i, (title, snippet, tagged, rows, result, cls) in enumerate(cases):
        x, y = 40 + 300 * i, 64
        out.append(f'<rect class="card" x="{x}" y="{y}" width="280" height="226" rx="10" stroke-width="1"/>')
        text(out, x + 16, y + 26, title, "th", "start")
        j = snippet.index(tagged)
        out.append(f'<rect class="tag" x="{x + 16 + j * CW - 2:g}" y="{y + 51}" width="{len(tagged) * CW + 4:g}" '
                   f'height="22" rx="4" stroke-width="1"/>')
        out.append(f'<text class="mono" x="{x + 16}" y="{y + 62}" dominant-baseline="central" '
                   f'textLength="{len(snippet) * CW:g}" lengthAdjust="spacingAndGlyphs" xml:space="preserve">{snippet}</text>')
        for k, (label, value) in enumerate(rows + [("Result", result)]):
            ry = y + 104 + 30 * k
            out.append(f'<line x1="{x + 16}" y1="{ry - 15}" x2="{x + 264}" y2="{ry - 15}" stroke="#E5E3DA" stroke-width="1"/>')
            text(out, x + 16, ry, label, "ts", "start")
            text(out, x + 264, ry, value, cls if label == "Result" else "tv", "end")
    write("tax_id_cases.svg", 960, 310, "How the scoring code treats a tax_id tag",
          "Three cases. A tax_id tag on a real NRIC is read as national_id and counts as correct. A tax_id tag on an "
          "NRIC-shaped value that has no gold span is read as national_id and counts as a false positive. "
          "A tax_id tag on a US ID number is not NRIC-shaped, stays tax_id, and counts as the wrong type.", out)


def harness():
    out = []
    box(out, 40, 165, 190, 80, "Opus 5.5", ["writes text only,", "has no tools"], "c-purple")
    out.append('<path d="M232 192 L318 192" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M318 222 L232 222" class="arr" marker-end="url(#arrow)"/>')
    text(out, 275, 178, "reply", "tl")
    text(out, 275, 238, "result", "tl")
    box(out, 320, 150, 200, 110, "Driver", ["takes one block per reply,", "runs it, sends back", "the result, logs it all"])
    out.append('<rect class="dr" x="600" y="10" width="330" height="220" rx="10" stroke-width="1.5"/>')
    text(out, 616, 28, "Walled-off area, set up by srt", "th", "start")
    text(out, 616, 45, "no network, no GPU, only the folders listed below", "tl", "start")
    box(out, 620, 56, 290, 70, "Python kernel", ["runs ```python blocks", "save_edit checks the edit rules"])
    for x, name, sub in ((620, "model/", "read only"), (720, "practice/", "read only"), (820, "work/", "read and write")):
        box(out, x, 146, 90, 60, name, [sub], "c-teal")
    box(out, 620, 262, 290, 90, "Forward-pass server", ["the only way MiniLM runs", "re-checks every edit in work/",
                                                        "scores with wsp.score"], "c-purple")
    out.append('<path d="M522 182 L618 96" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M522 238 L618 300" class="arr" marker-end="url(#arrow)"/>')
    text(out, 548, 122, "```python", "tl")
    text(out, 548, 288, "```forward", "tl")
    out.append('<path d="M865 260 L865 208" class="arr" marker-end="url(#arrow)" marker-start="url(#arrow)"/>')
    text(out, 857, 244, "reads edits, writes results", "tl", "end")
    box(out, 620, 376, 290, 56, "Main eval and regression check", ["hidden: nothing above can open them"], "c-coral")
    ly = 412
    for x, cls, label in ((40, "c-teal", "Opus 5.5's code can reach"), (240, "c-coral", "Hidden from Opus 5.5")):
        out.append(f'<rect class="{cls}" x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/>')
        text(out, x + 22, ly, label, "ts", "start")
    text(out, 40, 462, "Same workflow as the paper's Experiment 8, where Astra ran through the Codex CLI with its tools off.",
         "th", "start")
    text(out, 40, 482, "The paper did not publish its harness code, so this code is our rebuild of that workflow.",
         "ts", "start")
    write("harness.svg", 960, 504, "How a run works",
          "This is the same workflow as the paper's Experiment 8, rebuilt because the paper did not publish its "
          "harness code. "
          "Opus 5.5 writes text only. The driver takes one block from each reply, runs it, and sends back the result. "
          "A python block runs in the Python kernel inside a walled-off area, set up by srt, where code can reach only "
          "the folders listed for it. It can read model/ and practice/ and write work/, and nothing else. A forward "
          "block goes to the forward-pass server outside the walled-off area, the only way MiniLM runs. It re-checks "
          "the edits in work/, runs MiniLM, and scores with wsp.score. The main "
          "eval and the regression check are hidden.", out)


def edit_checks():
    out = []
    flow = [(40, 170, "Opus 5.5's code", ["in the kernel"], "box"),
            (270, 190, "save_edit", ["checks the edit first,", "refuses with a message"], "c-purple"),
            (520, 150, "work/edited/", ["one file per tensor"], "c-teal"),
            (730, 190, "Server checks again", ["before every forward pass", "and at the final answer"], "c-purple")]
    for x, w, title, lines, cls in flow:
        box(out, x, 40, w, 76, title, lines, cls)
    for x0, x1 in ((212, 268), (462, 518), (672, 728)):
        out.append(f'<path d="M{x0} 78 L{x1} 78" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M125 118 L125 152 L595 152 L595 120" class="arr" stroke-dasharray="5 4" '
               'marker-end="url(#arrow)"/>')
    text(out, 360, 166, "code that writes a file directly skips save_edit, so the server catches it", "tl")
    text(out, 40, 204, "Both checks refuse the same three kinds of edit:", "th", "start")
    rules = [("Embeddings", ["any tensor named *embeddings*"]),
             ("Classification head", ["classifier.weight, classifier.bias"]),
             ("Whole-tensor multiply", ["one scale factor explains 99%", "or more of the change"])]
    for i, (title, lines) in enumerate(rules):
        box(out, 40 + 300 * i, 222, 280, 76, title, lines, "c-coral")
    write("edit_checks.svg", 960, 318, "Where the edit rules are checked",
          "Opus 5.5's code saves an edit with save_edit, which checks it first. The edit goes to work/edited/. The "
          "forward-pass server checks every file there again before each forward pass and at the final answer, "
          "so code that writes a file directly and skips save_edit is still caught. Both checks refuse edits "
          "to the embeddings, edits to the classification head, and edits that one scale factor explains 99% or more "
          "of.", out)


def lock_checks():
    out = []
    text(out, 40, 22, "What lock.json freezes", "th", "start")
    groups = [("sets", ["the three sets", "and sets.json"]),
              ("scoring", ["score.py, predict.py,", "data.py"]),
              ("forward-pass server", ["server.py, edit.py,", "kernel.py, box.py"]),
              ("prompt", ["template, prompt.py,", "the 50-pass prompt"]),
              ("model", ["MiniLM's six files"]),
              ("environment", ["uv.lock, the srt pin,", "the Node version"]),
              ("baseline", ["baseline.py, outcomes.py,", "the prediction"]),
              ("the run's settings", ["CLI command and version,", "CLI environment, limits,", "no limit on turns"])]
    for i, (title, lines) in enumerate(groups):
        box(out, 40 + 225 * (i % 4), 40 + 92 * (i // 4), 210, 76, title, lines, "c-purple")
    out.append('<path d="M480 210 L480 240" class="arr" marker-end="url(#arrow)"/>')
    box(out, 330, 242, 300, 56, "lock.json", ["one hash per file, plus the run's settings"])
    text(out, 40, 336, "Where the files are checked against it", "th", "start")
    checks = [("M5 baseline", ["scripts/baseline.py", "refuses to score"], "c-teal"),
              ("M6 editor run", ["scripts/run_editor.py also checks", "the run's settings and the prompt"], "c-teal"),
              ("M7 evaluation", ["the edited model is", "scored the same way"], "box")]
    for i, (title, lines, cls) in enumerate(checks):
        x = 40 + 310 * i
        box(out, x, 390, 260, 76, title, lines, cls)
        out.append(f'<path d="M480 300 L{x + 130} 388" class="arr" marker-end="url(#arrow)"/>')
    text(out, 40, 496, "Any changed or missing file, or another Node version, stops the step with a list of what "
         "differs.", "ts", "start")
    ly = 528
    for x, cls, label in ((40, "c-teal", "Built and run in M5"), (240, "box", "Runs later")):
        out.append(f'<g class="{cls}"><rect x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/></g>')
        text(out, x + 22, ly, label, "ts", "start")
    write("lock_checks.svg", 960, 550, "What lock.json freezes and where it is checked",
          "Seven groups of files, the sets, the scoring code, the forward-pass server, the prompt, MiniLM's files, the "
          "environment and the baseline code with the prediction, are hashed once into lock.json, together with the "
          "run's settings: the exact CLI command, its version and environment, and the limits. The M5 baseline, the "
          "M6 editor run and the M7 evaluation each compare the files with lock.json and stop if any file changed. The M6 run also "
          "checks its settings and re-renders its prompt.", out)


def set_roles():
    out = []
    xs, xm, xr = 40, 330, 650

    def arrow(x0, y0, x1, y1, dashed=False):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<path d="M{x0} {y0} L{x1} {y1}" class="arr"{dash} marker-end="url(#arrow)"/>')

    m5 = 'fill="none" stroke="#2C2C2A" stroke-width="1.5" stroke-dasharray="6 4"'
    out.append(f'<rect x="636" y="34" width="308" height="458" rx="12" {m5}/>')
    text(out, 650, 54, "M5 freezes and scores these three sets", "th", "start")

    text(out, xs, 22, "Nemotron-PII: MiniLM's own data, made-up US personas", "th", "start")
    box(out, xs, 74, 230, 60, "Nemotron-PII", ["200,000 rows"])
    box(out, xm, 40, 250, 60, "Training split, 180,000 rows", ["MiniLM was fine-tuned on these", ], "c-purple")
    box(out, xm, 112, 250, 60, "Held-out split, 20,000 rows", ["MiniLM never trained on these"])
    arrow(272, 96, 328, 72)
    arrow(272, 112, 328, 140)
    box(out, xr, 76, 280, 96, "Regression check", ["18,515 rows that fit in 512 tokens",
                                                    "test set: did the edit break US text?",
                                                    ("M5 baseline F1 0.952, all labels", "tv")], "c-coral")
    arrow(582, 142, 648, 142)
    text(out, 616, 130, "filter", "tl")

    text(out, xs, 222, "openpii: Singapore rows, MiniLM never saw any", "th", "start")
    box(out, xs, 252, 230, 60, "openpii train split", ["20,225 Singapore rows"])
    box(out, xm, 252, 250, 60, "20,211 usable rows", ["filter drops long, duplicate rows", ])
    arrow(272, 282, 328, 282)
    box(out, xr, 230, 280, 108, "Practice set", ["1,000 rows, a random sample",
                                                 "Opus 5.5 reads the rows and",
                                                 "answers and scores its edits",
                                                 ("M5 baseline F1 0.682", "tv")], "c-teal")
    arrow(582, 282, 648, 282)
    text(out, 616, 270, "1,000", "tl")
    out.append(f'<rect class="dr" x="{xm}" y="338" width="250" height="34" rx="8" stroke-width="1"/>')
    text(out, xm + 125, 355, "The other 19,211 rows are not used", "ts")
    arrow(455, 314, 455, 336, dashed=True)

    box(out, xs, 402, 230, 60, "openpii validation split", ["5,005 Singapore rows"])
    box(out, xm, 402, 250, 60, "5,004 usable rows", ["filter drops 1 duplicate"])
    arrow(272, 432, 328, 432)
    box(out, xr, 384, 280, 96, "Main eval", ["all 5,004 rows",
                                             "test set: did the edit help Singapore?",
                                             ("M5 baseline F1 0.674", "tv")], "c-coral")
    arrow(582, 432, 648, 432)

    text(out, xs, 518, "Nothing in this experiment trains on the practice set, or on anything else. Opus 5.5 edits MiniLM's "
         "weights by hand.", "ts", "start")
    text(out, xs, 538, "The filter also drops every train row whose text or template appears in validation, so no "
         "practice row copies a main-eval row.", "ts", "start")
    ly = 572
    for x, cls, label in ((40, "c-purple", "Trained MiniLM, before this experiment"),
                          (330, "c-teal", "Opus 5.5 can read, rows and answers"),
                          (630, "c-coral", "Hidden test sets: Opus 5.5 never sees them")):
        out.append(f'<g class="{cls}"><rect x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/></g>')
        text(out, x + 22, ly, label, "ts", "start")
    out.append(f'<rect x="40" y="{ly + 19}" width="14" height="14" rx="3" {m5}/>')
    text(out, 62, ly + 26, "M5's part: these sets are hashed into lock.json, and the unedited MiniLM is scored on each. "
         "Everything outside it existed before M5.", "ts", "start")
    write("set_roles.svg", 960, ly + 48, "Where the three sets come from and what M5 does with them",
          "Two datasets feed the experiment. Nemotron-PII has 200,000 rows of made-up US personas. MiniLM was "
          "fine-tuned on 180,000 of them before this experiment. The other 20,000 were held out, and the 18,515 that "
          "fit in 512 tokens are the regression check, a hidden test set for US text. openpii's Singapore rows were "
          "never seen by MiniLM. From the train split's 20,211 usable rows, a random 1,000 form the practice set, "
          "which Opus 5.5 reads with answers and uses to score its edits. Nothing trains on it. The other 19,211 "
          "are not used. The validation split's 5,004 usable rows are the main eval, the hidden Singapore test set. "
          "A dotted line surrounds the three sets, the part M5 works on: M5 hashes them into lock.json and scores the "
          "unedited MiniLM on each, F1 0.952 on the regression check over all labels, 0.682 on the practice set and "
          "0.674 on the main eval. Nothing is trained in the experiment.", out)


def swatch(out, x, y, cls):
    out.append(f'<g class="{cls}"><rect x="{x}" y="{y - 7}" width="14" height="14" rx="3" stroke-width="0.5"/></g>')


def baseline_f1():
    out = []
    x0, bw = 330, 520

    def f1_pair(y, sg, nem):
        for dy, f1, cls in ((0, sg, "c-coral"), (20, nem, "box")):
            out.append(f'<g class="{cls}"><rect x="{x0}" y="{y + dy}" width="{bw * f1:g}" height="16" rx="3" '
                       f'stroke-width="0.5"/></g>')
            text(out, x0 + bw * f1 + 8, y + dy + 8, f"{f1:.3f}", "tv", "start")

    text(out, 40, 22, "Baseline: F1 of the unedited MiniLM on each scored label", "th", "start")
    text(out, 40, 44, "Measured in M5, before any edit. Each pair compares Singapore test text with test text like "
         "MiniLM's own training data.", "ts", "start")
    for x, cls, label in ((40, "c-coral", "Singapore test set (main eval), 5,004 rows"),
                          (400, "box", "Nemotron held-out test set, 18,515 rows")):
        swatch(out, x, 74, cls)
        text(out, x + 22, 74, label, "ts", "start")
    labels = [("name", "given names and surnames, 11,940 Singapore entities", 0.8739, 0.9889),
              ("national_id", "1,034 Singapore entities", 0.7374, 0.959)]
    for i, (name, sub, sg, nem) in enumerate(labels):
        y = 100 + 52 * i
        text(out, 40, y + 8, name, "mono", "start")
        text(out, 40, y + 26, sub, "tl", "start")
        f1_pair(y, sg, nem)
    out.append('<path d="M40 206 L900 206" stroke="#E5E3DA" stroke-width="1"/>')
    text(out, 40, 222, "Both labels counted together", "th", "start")
    text(out, 40, 240, "used by the check that there", "tl", "start")
    text(out, 40, 254, "is something to repair", "tl", "start")
    f1_pair(218, 0.8649, 0.9882)
    text(out, 40, 280, "Counted together, every entity counts once. Names are 92% of the entities, so they pull F1 above "
         "the mean of the two labels, 0.806.", "ts", "start")
    text(out, 40, 298, "Gap 0.123. The check that there is something to repair needs more than twice its standard error, "
         "2 × 0.0030 = 0.0060. The check passes.", "ts", "start")
    write("baseline_f1.svg", 960, 320, "Baseline F1 of the unedited MiniLM by scored label",
          "F1 of the unedited MiniLM, measured in M5, on each scored label, Singapore test set against Nemotron "
          "held-out test set. On Singapore text given names and surnames are scored as one label, name: name 0.874 "
          "against 0.989, national_id 0.737 against 0.959. With both labels counted together, which is what the check "
          "that there is something to repair compares, Singapore scores 0.865 and Nemotron 0.988, a gap of 0.123, far "
          "above the check's threshold of 0.0060. The check passes.", out)


def baseline_kinds():
    out = []
    x0, bw = 390, 480
    text(out, 40, 22, "Baseline: which Singapore entities the unedited MiniLM gets right, by category", "th", "start")
    text(out, 40, 44, "Singapore test set (the main eval, 5,004 rows). Each bar is the share of one category's entities that "
         "MiniLM tags exactly right.", "ts", "start")
    text(out, 40, 62, "Given names and surnames are scored as one label, name. One category is almost never right, and "
         "the prediction says the edit will fix it.", "ts", "start")
    text(out, 40, 80, '"Just before" means within the 60 characters before the number, about ten words.', "ts", "start")
    groups = [("national_id", [('NRIC without "tax" just before it', 323, 0.0, True,
                                "most tagged biometric_identifier (168) or missed (118)"),
                               ('NRIC with "tax" just before it', 711, 0.8748, False, "")]),
              ("name, given names", [("Given name of one word", 3376, 0.8451, False, ""),
                                     ("Given name of several words, first word", 1434, 0.8717, False, ""),
                                     ("Given name of several words, later words", 1434, 0.9073, False, "")]),
              ("name, surnames", [("Surname of one word", 2688, 0.923, False, ""),
                                  ("Surname of several words, first word", 1274, 0.8689, False, ""),
                                  ("Surname of several words, later words", 1734, 0.8316, False, "")])]
    y = 102
    for label, categories in groups:
        text(out, 40, y + 10, label, "mono", "start")
        y += 26
        for name, n, share, fixed, note in categories:
            text(out, 56, y + 10, f"{name} ({n:,})", "ts", "start")
            cls = "c-teal" if fixed else "box"
            out.append(f'<rect x="{x0}" y="{y}" width="{bw}" height="20" rx="3" fill="#FAFAF7" '
                       f'stroke="{"#0F6E56" if fixed else "#E5E3DA"}" stroke-width="{1 if fixed else 0.5}"/>')
            if share:
                out.append(f'<g class="{cls}"><rect x="{x0}" y="{y}" width="{bw * share:g}" height="20" rx="3" '
                           f'stroke-width="0.5"/></g>')
            if note:
                text(out, x0 + bw * share + 10, y + 10, note, "tl", "start")
            text(out, x0 + bw + 10, y + 10, f"{share:.0%}", "tv", "start")
            y += 30
        y += 6
    ly = y + 10
    for x, cls, label in ((40, "c-teal", "Predicted fixed by the edit"), (280, "box", "Predicted unchanged")):
        swatch(out, x, ly, cls)
        text(out, x + 22, ly, label, "ts", "start")
    write("baseline_kinds.svg", 960, ly + 22, "Baseline share right by category of Singapore entity",
          "The share of entities the unedited MiniLM tags with the exact span and the right label on the Singapore test "
          "set, the main eval, split by category. Given names and surnames are scored as one label, name. Just before "
          'means within the 60 characters before the number. The category "NRIC without "tax" just before it" is 0% '
          "right, most tagged biometric_identifier or missed, and is the one category predicted to be fixed. The other "
          "categories are 83% to 92% right.", out)


def run_timeline():
    out = []
    x0, x1, last = 130, 890, 232

    def tx(turn):
        return x0 + (x1 - x0) * turn / last

    passes = {"dev": [6, 29, 43, 55, 65, 73, 79, 90, 96, 102, 108, 117, 141, 147, 164, 178, 181, 214],
              "hold": [121, 153, 156, 158, 160, 166, 168, 208, 223, 231],
              "texts": [19, 49, 130, 132, 134, 138, 169, 176, 184, 188, 192, 196, 198, 201, 206, 209, 212, 215, 217,
                        221, 225, 228]}
    lanes = [("dev", "dev half", "c-teal", 18), ("hold", "held-back half", "c-purple", 10),
             ("texts", "own texts", "box", 22)]
    text(out, 40, 22, "Run p50: when Opus 5.5 spent its 50 forward passes, and how its context grew", "th", "start")
    text(out, 40, 42, "Each mark is one forward pass, placed at the turn that asked for it, in the lane for what MiniLM "
         "ran on.", "ts", "start")
    for i, (key, label, cls, n) in enumerate(lanes):
        y = 70 + 24 * i
        text(out, 40, y + 8, label, "ts", "start")
        text(out, x1 + 12, y + 8, f"{n} passes", "tl", "start")
        out.append(f'<line x1="{x0}" y1="{y + 8}" x2="{x1}" y2="{y + 8}" stroke="#E5E3DA" stroke-width="1"/>')
        for t in passes[key]:
            out.append(f'<g class="{cls}"><rect x="{tx(t) - 2:g}" y="{y}" width="4" height="16" rx="1" '
                       f'stroke-width="0.5"/></g>')

    top, bot, ymax = 190, 350, 1_000_000

    def ty(tokens):
        return bot - (bot - top) * tokens / ymax

    events = [(43, "turn 43: first edit tested", 150), (208, "turn 208: final edit first tested", 150),
              (232, "turn 232: final answer", 166)]
    for t, label, ly in events:
        x = tx(t)
        for ya, yb in ((64, 140), (176, bot)):
            out.append(f'<line x1="{x:g}" y1="{ya}" x2="{x:g}" y2="{yb}" stroke="#888780" stroke-width="1" '
                       f'stroke-dasharray="2 3"/>')
        text(out, x - 4, ly, label, "tl", "end")
    for tokens, label in ((0, "0"), (500_000, "500k"), (1_000_000, "1M")):
        y = ty(tokens)
        out.append(f'<line x1="{x0}" y1="{y:g}" x2="{x1}" y2="{y:g}" stroke="#E5E3DA" stroke-width="1"/>')
        text(out, x0 - 8, y, label, "tl", "end")
    text(out, 40, (top + bot) / 2 - 8, "context,", "ts", "start")
    text(out, 40, (top + bot) / 2 + 8, "tokens", "ts", "start")
    yc = ty(967_000)
    out.append(f'<line x1="{x0}" y1="{yc:g}" x2="{x1}" y2="{yc:g}" stroke="#A32D2D" stroke-width="1" '
               f'stroke-dasharray="5 4"/>')
    text(out, x0 + 4, yc + 12, "the CLI compacts the conversation above 967,000 tokens", "bad", "start")
    ctx = [(1, 4), (20, 57), (40, 136), (60, 219), (80, 281), (100, 349), (120, 418), (140, 489), (160, 554),
           (180, 629), (200, 681), (220, 751), (232, 780)]
    pts = " ".join(f"{tx(t):g},{ty(k * 1000):g}" for t, k in ctx)
    out.append(f'<polyline points="{pts}" fill="none" stroke="#534AB7" stroke-width="2"/>')
    text(out, tx(232) + 8, ty(780_000), "780,000", "tv", "start")
    for t in (0, 50, 100, 150, 200):
        text(out, tx(t), bot + 16, str(t), "tl")
    text(out, (x0 + x1) / 2, bot + 34, "turn", "ts")
    notes = ["dev half: the 500 practice rows Opus 5.5 used to build and tune its edits.",
             "held-back half: the other 500 practice rows, which Opus 5.5 set aside to test its edits.",
             "own texts: sentences Opus 5.5 wrote, such as US-style checks and unseen NRIC contexts."]
    for i, s in enumerate(notes):
        text(out, 40, bot + 64 + 18 * i, s, "ts", "start")
    write("run_timeline.svg", 960, bot + 112, "When run p50 spent its forward passes, and how its context grew",
          "Run p50 lasted 232 turns. Its 50 forward passes ran MiniLM on three kinds of input: 18 on the dev half of "
          "the practice set, 10 on the held-back half, and 22 on texts Opus 5.5 wrote. The first edit was tested at "
          "turn 43, the final edit was first tested at turn 208, and the final answer came at turn 232. The context "
          "grew steadily to 780,000 tokens and never reached 967,000, where the CLI compacts the conversation.", out)


def edit_map():
    out = []
    text(out, 40, 22, "Where the submitted edit changes MiniLM", "th", "start")
    text(out, 40, 42, "MiniLM's six layers, bottom to top in the order a text passes through them. Each layer has an "
         "attention block and an MLP of 1,536 neurons.", "ts", "start")
    xa, xm, xn = 130, 320, 570
    head_y, row_h = 64, 56
    box(out, xa, head_y, 420, 44, "Classification head", ["edits refused by save_edit"], "c-coral")
    mlps = {5: ("MLP", "tried, not submitted", "dashed",
                ["Hyphenated names, neurons 161, 423, 541, 665.",
                 "Dropped: gain on the held-back half +0.014 ± 0.015."]),
            4: ("MLP", "2 neurons edited", "c-teal",
                ["NRIC repair, neurons 806 and 1407.",
                 "They detect the NRIC shape and write the national-ID state."]),
            2: ("MLP", "4 neurons edited", "c-teal",
                ["Title repair, neurons 792 and 1209, plus 244 and 1258",
                 "for Mayoress. They write the signal Mr and Mrs carry."]),
            1: ("MLP", "2 neurons edited", "c-teal",
                ["Title repair, neurons 54 and 485.",
                 "They detect Singapore titles such as Mstr and Madame."])}
    for i, layer in enumerate(range(5, -1, -1)):
        y = head_y + 60 + row_h * i
        text(out, 40, y + 20, f"Layer {layer}", "ts", "start")
        box(out, xa, y, 170, 40, "Attention", ["unchanged"])
        title, sub, cls, note = mlps.get(layer, ("MLP", "unchanged", "box", []))
        if cls == "dashed":
            out.append(f'<rect class="dr" x="{xm}" y="{y}" width="230" height="40" rx="8" stroke-width="1"/>')
            text(out, xm + 115, y + 12, title, "th")
            text(out, xm + 115, y + 30, sub, "ts")
        else:
            box(out, xm, y, 230, 40, title, [sub], cls)
        for j, s in enumerate(note):
            text(out, xn, y + 12 + 16 * j, s, "tl" if cls == "dashed" else "ts", "start")
    emb_y = head_y + 60 + row_h * 6
    box(out, xa, emb_y, 420, 44, "Embeddings", ["edits refused by save_edit"], "c-coral")
    out.append(f'<path d="M110 {emb_y + 40} L110 {head_y + 6}" class="arr" marker-end="url(#arrow)"/>')
    text(out, 40, emb_y + 72, "In each edited MLP the edit rewrites those neurons' input weights, input biases and output weights, and "
         "shifts the layer's output bias", "ts", "start")
    text(out, 40, emb_y + 90, "to cancel the neurons' old average output. 12 tensors and 7,304 numbers change. Every "
         "other number in MiniLM is unchanged.", "ts", "start")
    ly = emb_y + 122
    for x, cls, label in ((40, "c-teal", "Edited in the submitted answer"), (300, "box", "Unchanged"),
                          (460, "c-coral", "Edits not allowed")):
        swatch(out, x, ly, cls)
        text(out, x + 22, ly, label, "ts", "start")
    out.append(f'<rect class="dr" x="640" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="1"/>')
    text(out, 662, ly, "Tried during the run, not submitted", "ts", "start")
    write("edit_map.svg", 960, ly + 24, "Where the submitted edit changes MiniLM",
          "MiniLM's six layers, each with an attention block and an MLP. The submitted edit changes only the MLPs of "
          "layers 1, 2 and 4. Layers 1 and 2 hold the title repair: neurons 54 and 485 in layer 1, and 792, 1209, 244 "
          "and 1258 in layer 2. Layer 4 holds the NRIC repair, neurons 806 and 1407. A repair for hyphenated names in "
          "layer 5 was tried and dropped. Every attention block is unchanged, and edits to the embeddings and the "
          "classification head are not allowed. In all, 12 tensors and 7,304 numbers change.", out)


def evaluation():
    return json.loads((HERE.parents[1] / "runs" / "p50" / "eval" / "evaluation.json").read_text())


KIND_NAMES = {"national_id/no tax context": 'NRIC without "tax" just before it',
              "national_id/tax context": 'NRIC with "tax" just before it',
              "first_name/one word": "Given name of one word",
              "first_name/multi-word, first word": "Given name of several words, first word",
              "first_name/multi-word, later word": "Given name of several words, later words",
              "last_name/one word": "Surname of one word",
              "last_name/multi-word, first word": "Surname of several words, first word",
              "last_name/multi-word, later word": "Surname of several words, later words"}


def m7_kinds():
    verdicts = evaluation()["kinds"]["verdicts"]
    out = []
    x0, bw = 360, 400
    text(out, 40, 22, "Which Singapore entities MiniLM gets right, before and after the edit, by category", "th", "start")
    text(out, 40, 44, "Main eval, 5,004 rows. Each pair is the share of one category's entities tagged with the exact "
         "span and the right label.", "ts", "start")
    text(out, 40, 62, "Fixed means a rise of at least 10 points. The last two columns give the M5 prediction and the "
         "result under that rule.", "ts", "start")
    text(out, x0 + bw + 80, 88, "Predicted", "tl")
    text(out, x0 + bw + 160, 88, "Result", "tl")
    groups = [("national_id", list(KIND_NAMES)[:2]), ("name, given names", list(KIND_NAMES)[2:5]),
              ("name, surnames", list(KIND_NAMES)[5:])]
    y = 100
    for label, kinds in groups:
        text(out, 40, y + 10, label, "mono", "start")
        y += 26
        for k in kinds:
            v = verdicts[k]
            predicted = "fixed" if k == "national_id/no tax context" else "unchanged"
            text(out, 56, y + 16, f"{KIND_NAMES[k]} ({v['entities']:,})", "ts", "start")
            for dy, share, cls in ((0, v["base_share"], "box"), (16, v["edited_share"], "c-teal")):
                out.append(f'<rect x="{x0}" y="{y + dy}" width="{bw}" height="14" rx="3" fill="#FAFAF7" '
                           f'stroke="#E5E3DA" stroke-width="0.5"/>')
                if share:
                    out.append(f'<g class="{cls}"><rect x="{x0}" y="{y + dy}" width="{bw * share:g}" height="14" '
                               f'rx="3" stroke-width="0.5"/></g>')
                text(out, x0 + bw * share + 6, y + dy + 7, f"{share:.0%}", "tl", "start")
            text(out, x0 + bw + 80, y + 15, predicted, "ts")
            text(out, x0 + bw + 160, y + 15, v["verdict"], "ok" if v["verdict"] == predicted else "bad")
            y += 40
        y += 4
    note = verdicts["national_id/tax context"]
    text(out, 40, y + 4, f'NRICs with "tax" just before them rose {note["change_pp"]:.2f} points, '
         f'{note["base_correct"]} to {note["edited_correct"]} of {note["entities"]}, just under the line for fixed.',
         "ts", "start")
    ly = y + 30
    for x, cls, lab in ((40, "box", "Unedited MiniLM"), (220, "c-teal", "Edited MiniLM")):
        swatch(out, x, ly, cls)
        text(out, x + 22, ly, lab, "ts", "start")
    write("m7_kinds.svg", 960, ly + 22, "Share right by category, before and after the edit",
          "The share of Singapore main-eval entities MiniLM tags exactly right, before and after Opus 5.5's edit, by "
          'category. NRICs without "tax" just before them rise from 0% to 85%, which the prediction called fixed. '
          'NRICs with "tax" just before them rise from 87% to 97%, just under the 10-point line. Every name category '
          "rises by 1 to 3 points, which counts as unchanged, as predicted.", out)


def m7_errors():
    ev = evaluation()
    out = []
    text(out, 40, 22, "Where the Singapore errors went, and what the edit cost on Nemotron", "th", "start")
    text(out, 40, 44, "Left: errors on the Singapore main eval by kind. Right: the Nemotron labels whose F1 moved most, "
         "out of all 55.", "ts", "start")
    base, edited = ev["scores"]["singapore"]["base"]["failures"], ev["scores"]["singapore"]["edited"]["failures"]
    x0, scale = 170, 250 / 1200
    names = {"false_positive": "False positive", "wrong_type": "Wrong type", "missed": "Missed",
             "wrong_boundary": "Wrong boundary"}
    text(out, 40, 76, "Singapore errors", "th", "start")
    y = 96
    for k, lab in names.items():
        text(out, 40, y + 15, lab, "ts", "start")
        for dy, n, cls in ((0, base[k], "box"), (16, edited[k], "c-teal")):
            out.append(f'<g class="{cls}"><rect x="{x0}" y="{y + dy}" width="{n * scale:g}" height="14" rx="3" '
                       f'stroke-width="0.5"/></g>')
            text(out, x0 + n * scale + 6, y + dy + 7, f"{n:,}", "tl", "start")
        y += 42
    titles = ev["mechanism"]["titles"]["groups"]["targeted"]
    text(out, 40, y + 6, f"Titles tagged as a name fell from {titles['base_hit']} to {titles['edited_hit']},", "tl", "start")
    text(out, 40, y + 22, "most of the drop in false positives.", "tl", "start")

    rows = ev["nemotron_regression_by_label"]
    shown = rows[:4] + rows[-2:]
    x1, mid, dscale = 520, 880, 120 / 0.05
    text(out, x1, 76, "Nemotron F1 change, edited minus unedited", "th", "start")
    out.append(f'<path d="M{mid} 92 L{mid} {96 + 30 * len(shown)}" stroke="#888780" stroke-width="1"/>')
    for i, r in enumerate(shown):
        yy = 100 + 30 * i
        text(out, x1, yy + 8, r["label"], "mono", "start")
        w = r["change"] * dscale
        cls = "c-coral" if w < 0 else "c-teal"
        out.append(f'<g class="{cls}"><rect x="{mid + min(w, 0):g}" y="{yy}" width="{abs(w):g}" height="16" rx="3" '
                   f'stroke-width="0.5"/></g>')
        text(out, mid + max(w, 0) + 6, yy + 8, f"{r['change']:+.3f}", "tl", "start")
    t = ev["nemotron_transitions"][0]
    yy = 104 + 30 * len(shown)
    total = ev["scores"]["nemotron_regression"]["bootstrap"]
    text(out, x1, yy + 6, f"{t['shapes']['A99999999999']} biometric IDs shaped like a letter and 11 digits are now tagged national_id.",
         "tl", "start")
    text(out, x1, yy + 22, f"All 55 labels together: {total['diff']:+.4f} ± {total['diff_se']:.4f}, "
         "inside the prediction's 0.005.", "tl", "start")
    ly = max(y, yy) + 52
    for x, cls, lab in ((40, "box", "Unedited MiniLM"), (220, "c-teal", "Edited MiniLM, or a gain"),
                        (440, "c-coral", "A loss")):
        swatch(out, x, ly, cls)
        text(out, x + 22, ly, lab, "ts", "start")
    write("m7_errors.svg", 960, ly + 22, "Singapore errors by kind and Nemotron F1 changes by label",
          "Left: errors on the Singapore main eval by kind, before and after the edit. False positives fall from "
          "1,196 to 251, most of them titles no longer tagged as names. Wrong types fall from 610 to 339, mostly "
          "NRICs. Right: the Nemotron labels whose F1 changed most. national_id falls by 0.047 and "
          "biometric_identifier by 0.014, because biometric IDs shaped like a letter and 11 digits are now tagged "
          "national_id. Over all 55 labels, Nemotron F1 changes by -0.0004.", out)


data_sets()
example_rows()
label_translation()
tax_id_cases()
harness()
edit_checks()
lock_checks()
set_roles()
baseline_f1()
baseline_kinds()
run_timeline()
edit_map()
m7_kinds()
m7_errors()

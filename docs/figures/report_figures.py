"""Generate the figures in the M2 to M5 reports: data_sets, example_rows, label_translation, tax_id_cases,
harness, edit_checks, lock_checks, set_roles, and baseline_kinds."""

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
        text(out, x + w / 2, top + 18 + 16 * i, ln, "ts")
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
         "drops long, duplicate, and shared texts", "Practice set", ["1,000 rows", "editor sees rows and answers"], "c-teal"),
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
    for x, cls, label in ((40, "c-teal", "Editor can see"), (200, "c-coral", "Hidden from editor")):
        out.append(f'<rect class="{cls}" x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/>')
        text(out, x + 22, ly, label, "ts", "start")
    write("data_sets.svg", 960, 352, "Where each set comes from",
          "Three sources pass a filter into three sets. openpii train becomes the 1,000-row practice set, the only set "
          "the editor sees. openpii validation becomes the 5,004-row main eval, and Nemotron's held-out split becomes "
          "the 18,515-row regression check. Both are hidden from the editor.", out)


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
    text(out, x0 + 22, ly, "Scored. openpii's label → the model's label", "ts", "start")
    out.append(f'<rect class="dr" x="{x0 + 330}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="1"/>')
    text(out, x0 + 352, ly, "Not scored. openpii's label", "ts", "start")
    write("example_rows.svg", 960, 384, "Two main-eval rows and their answer keys",
          "Two openpii rows with every answer-key span marked. Given names, surnames, and the NRIC are scored, "
          "renamed to first_name, last_name, and national_id. Titles, dates, cities, addresses, and the passport "
          "and ID card numbers are not scored.", out)


def label_translation():
    out = []
    for x, s in ((140, "openpii answer key"), (440, "Answer key in the model's names"), (780, "Model output")):
        text(out, x, 24, s, "ts")
    for i, (src, sub, dst) in enumerate([("GIVENNAME", "given names", "first_name"), ("SURNAME", "surnames", "last_name"),
                                         ("TAXNUM", "every value an NRIC", "national_id")]):
        y = 44 + 72 * i
        box(out, 40, y, 200, 56, src, [sub])
        out.append(f'<path d="M242 {y + 28} L336 {y + 28}" class="arr" marker-end="url(#arrow)"/>')
        box(out, 340, y, 200, 56, dst, cls="c-purple")
    box(out, 640, 44, 280, 200, "Model's tags",
        ["55 types, such as first_name,", "national_id, and tax_id", "", "The types never change.",
         "Only the weights are edited."])
    out.append('<path d="M440 246 L440 280" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M780 246 L780 280" class="arr" marker-end="url(#arrow)"/>')
    box(out, 340, 282, 580, 56, "Grader", ["A tag counts only if its words and its type match the answer key."], "c-purple")
    write("label_translation.svg", 960, 360, "The answer key is translated, not the model",
          "openpii's labels GIVENNAME, SURNAME, and TAXNUM are renamed to the model's labels first_name, last_name, "
          "and national_id. The model's 55 tag types never change. The grader compares the model's tags with the "
          "renamed answer key.", out)


def tax_id_cases():
    out = []
    text(out, 40, 22, "How the grader treats a tax_id tag", "th", "start")
    text(out, 40, 42, "The grader reads a tax_id tag as national_id only when the tagged text is NRIC-shaped, "
         "a letter, seven digits, and a letter.", "ts", "start")
    cases = [
        ("A real NRIC", "the tax details under G6346291N", "G6346291N",
         [("Model's tag", "tax_id"), ("Answer key", "national_id"), ("Grader reads", "national_id")],
         "correct", "ok"),
        ("An NRIC the key doesn't mark", "IDs S1234567D and T7654321A.", "T7654321A",
         [("Model's tag", "tax_id"), ("Answer key", "nothing here"), ("Grader reads", "national_id")],
         "mistake, a false positive", "bad"),
        ("A US ID number in Nemotron", "SSN 123-45-6789.", "123-45-6789",
         [("Model's tag", "tax_id"), ("Answer key", "national_id"), ("Grader reads", "tax_id, not NRIC-shaped")],
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
    write("tax_id_cases.svg", 960, 310, "How the grader treats a tax_id tag",
          "Three cases. A tax_id tag on a real NRIC is read as national_id and counts as correct. A tax_id tag on an "
          "NRIC-shaped value that the answer key does not mark is read as national_id and counts as a false positive. "
          "A tax_id tag on a US ID number is not NRIC-shaped, stays tax_id, and counts as the wrong type.", out)


def harness():
    out = []
    box(out, 40, 165, 190, 80, "Opus 5.5", ["writes text only,", "has no tools"], "c-purple")
    out.append('<path d="M232 192 L318 192" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M318 222 L232 222" class="arr" marker-end="url(#arrow)"/>')
    text(out, 275, 178, "reply", "tl")
    text(out, 275, 238, "result", "tl")
    box(out, 320, 150, 200, 110, "Driver", ["takes one block per reply,", "runs it, sends back", "the result, logs it all"])
    out.append('<rect class="dr" x="600" y="20" width="330" height="210" rx="10" stroke-width="1.5"/>')
    text(out, 616, 38, "srt sandbox", "th", "start")
    text(out, 712, 38, "no network, no GPU, no other files", "tl", "start")
    box(out, 620, 56, 290, 70, "Python kernel", ["runs ```python blocks", "save_edit checks the edit rules"])
    for x, name, sub in ((620, "model/", "read only"), (720, "practice/", "read only"), (820, "work/", "read and write")):
        box(out, x, 146, 90, 60, name, [sub], "c-teal")
    box(out, 620, 262, 290, 90, "Forward-pass server", ["the only way MiniLM runs", "re-checks every edit in work/",
                                                        "grades with wsp.score"], "c-purple")
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
          "A python block runs in the Python kernel inside the srt sandbox, which can read model/ and practice/ and "
          "write work/, and nothing else. A forward block goes to the forward-pass server outside the sandbox, the "
          "only way MiniLM runs. It re-checks the edits in work/, runs MiniLM, and grades with wsp.score. The main "
          "eval and the regression check are hidden.", out)


def edit_checks():
    out = []
    flow = [(40, 170, "Opus 5.5's code", ["in the kernel"], "box"),
            (270, 190, "save_edit", ["check 1: refuses with", "a message"], "c-purple"),
            (520, 150, "work/edited/", ["one file per tensor"], "c-teal"),
            (730, 190, "Server re-check", ["check 2: before every forward", "pass and at the final answer"], "c-purple")]
    for x, w, title, lines, cls in flow:
        box(out, x, 40, w, 76, title, lines, cls)
    for x0, x1 in ((212, 268), (462, 518), (672, 728)):
        out.append(f'<path d="M{x0} 78 L{x1} 78" class="arr" marker-end="url(#arrow)"/>')
    out.append('<path d="M125 118 L125 152 L595 152 L595 120" class="arr" stroke-dasharray="5 4" '
               'marker-end="url(#arrow)"/>')
    text(out, 360, 166, "code that writes a file directly skips check 1, so check 2 catches it", "tl")
    text(out, 40, 204, "Both checks refuse the same three kinds of edit:", "th", "start")
    rules = [("Embeddings", ["any tensor named *embeddings*"]),
             ("Classification head", ["classifier.weight, classifier.bias"]),
             ("Whole-tensor multiply", ["one scale factor explains 99%", "or more of the change"])]
    for i, (title, lines) in enumerate(rules):
        box(out, 40 + 300 * i, 222, 280, 76, title, lines, "c-coral")
    write("edit_checks.svg", 960, 318, "Where the edit rules are checked",
          "Opus 5.5's code saves an edit with save_edit, which is check 1. The edit goes to work/edited/. The "
          "forward-pass server re-checks every file there before each forward pass and at the final answer, which is "
          "check 2, so code that writes a file directly and skips save_edit is still caught. Both checks refuse edits "
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
              ("run settings", ["CLI command and version,", "limits, no turn cap"])]
    for i, (title, lines) in enumerate(groups):
        box(out, 40 + 225 * (i % 4), 40 + 92 * (i // 4), 210, 76, title, lines, "c-purple")
    out.append('<path d="M480 210 L480 240" class="arr" marker-end="url(#arrow)"/>')
    box(out, 330, 242, 300, 56, "lock.json", ["one SHA-256 per file, plus the run settings"])
    text(out, 40, 336, "Where the tree is checked against it", "th", "start")
    checks = [("M5 baseline", ["scripts/baseline.py", "refuses to score"], "c-teal"),
              ("M6 editor run", ["scripts/run_editor.py also checks", "the settings and the prompt"], "c-teal"),
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
          "run settings: the exact CLI command, the CLI version and the limits. The M5 baseline, the M6 editor run "
          "and the M7 evaluation each compare the tree with lock.json and stop if any file changed. The M6 run also "
          "checks its settings and re-renders its prompt.", out)


def set_roles():
    out = []
    xs, xm, xr = 40, 330, 650

    def arrow(x0, y0, x1, y1, dashed=False):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<path d="M{x0} {y0} L{x1} {y1}" class="arr"{dash} marker-end="url(#arrow)"/>')

    text(out, xs, 22, "Nemotron-PII: MiniLM's own data, made-up US personas", "th", "start")
    box(out, xs, 74, 230, 60, "Nemotron-PII", ["200,000 rows"])
    box(out, xm, 40, 250, 60, "Training split, 180,000 rows", ["MiniLM was fine-tuned on these", ], "c-purple")
    box(out, xm, 112, 250, 60, "Held-out split, 20,000 rows", ["MiniLM never trained on these"])
    arrow(272, 96, 328, 72)
    arrow(272, 112, 328, 140)
    box(out, xr, 104, 280, 76, "Regression check", ["18,515 rows that fit in 512 tokens",
                                                     "test set: did the edit break US text?"], "c-coral")
    arrow(582, 142, 648, 142)
    text(out, 616, 130, "filter", "tl")

    text(out, xs, 222, "openpii: Singapore rows, MiniLM never saw any", "th", "start")
    box(out, xs, 252, 230, 60, "openpii train split", ["20,225 Singapore rows"])
    box(out, xm, 252, 250, 60, "20,211 usable rows", ["filter drops long, duplicate rows", ])
    arrow(272, 282, 328, 282)
    box(out, xr, 236, 280, 92, "Practice set", ["1,000 rows, a random sample",
                                                "development set: Opus 5.5 reads",
                                                "the answers and grades its edits"], "c-teal")
    arrow(582, 270, 648, 270)
    text(out, 616, 258, "1,000", "tl")
    out.append(f'<rect class="dr" x="{xr}" y="340" width="280" height="34" rx="8" stroke-width="1"/>')
    text(out, xr + 140, 357, "The other 19,211 rows are not used", "ts")
    arrow(582, 296, 648, 352, dashed=True)

    box(out, xs, 402, 230, 60, "openpii validation split", ["5,005 Singapore rows"])
    box(out, xm, 402, 250, 60, "5,004 usable rows", ["filter drops 1 duplicate"])
    arrow(272, 432, 328, 432)
    box(out, xr, 394, 280, 76, "Main eval", ["all 5,004 rows",
                                             "test set: did the edit help Singapore?"], "c-coral")
    arrow(582, 432, 648, 432)

    text(out, xs, 506, "Nothing is trained in this experiment. Opus 5.5 edits MiniLM's weights by hand, so the practice "
         "set is its development set, not training data.", "ts", "start")
    text(out, xs, 526, "The filter also drops every train row whose text or template appears in validation, so no "
         "practice row copies a main-eval row.", "ts", "start")
    ly = 560
    for x, cls, label in ((40, "c-purple", "Trained MiniLM, before this experiment"),
                          (330, "c-teal", "Opus 5.5 can read, rows and answers"),
                          (630, "c-coral", "Hidden test sets, scored at M5 and M7")):
        out.append(f'<g class="{cls}"><rect x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/></g>')
        text(out, x + 22, ly, label, "ts", "start")
    write("set_roles.svg", 960, 582, "Where the three sets come from and what each one is for",
          "Two datasets feed the experiment. Nemotron-PII has 200,000 rows of made-up US personas. MiniLM was "
          "fine-tuned on 180,000 of them before this experiment. The other 20,000 were held out, and the 18,515 that "
          "fit in 512 tokens are the regression check, a hidden test set for US text. openpii's Singapore rows were "
          "never seen by MiniLM. From the train split's 20,211 usable rows, a random 1,000 form the practice set, "
          "which Opus 5.5 reads with answers and uses to grade its edits, like a development set. The other 19,211 "
          "are not used. The validation split's 5,004 usable rows are the main eval, the hidden Singapore test set. "
          "Nothing is trained in the experiment.", out)


def baseline_kinds():
    out = []
    x0, bw = 330, 520
    text(out, 40, 22, "The gate: F1 on the three scored labels", "th", "start")
    for i, (name, f1, cls) in enumerate((("Singapore main eval, 5,004 rows", 0.674, "c-coral"),
                                         ("Nemotron held-out, 18,515 rows", 0.984, "box"))):
        y = 48 + 34 * i
        text(out, 40, y + 11, name, "ts", "start")
        out.append(f'<g class="{cls}"><rect x="{x0}" y="{y}" width="{bw * f1:g}" height="22" rx="3" stroke-width="0.5"/></g>')
        text(out, x0 + bw * f1 + 8, y + 11, f"{f1:.3f}", "tv", "start")
    text(out, 40, 130, "Gap 0.310. The gate needs more than twice its standard error, 2 × 0.0044 = 0.0087. It passes.",
         "ts", "start")
    text(out, 40, 176, "Share of entities MiniLM gets right on the main eval, by kind", "th", "start")
    kinds = [("NRIC, no tax context", 323, 0.0, True), ("NRIC, tax context", 711, 0.8748, False),
             ("Given name, several words, later word", 1434, 0.0516, False),
             ("Given name, one word", 3376, 0.6182, False), ("Given name, several words, first word", 1434, 0.6667, False),
             ("Surname, one word", 2688, 0.8943, False), ("Surname, several words, first word", 1274, 0.8124, False),
             ("Surname, several words, later word", 1734, 0.8287, False)]
    for i, (name, n, share, fixed) in enumerate(kinds):
        y = 196 + 30 * i
        text(out, 40, y + 10, f"{name} ({n:,})", "ts", "start")
        cls = "c-teal" if fixed else "box"
        out.append(f'<rect x="{x0}" y="{y}" width="{bw}" height="20" rx="3" fill="#FAFAF7" '
                   f'stroke="{"#0F6E56" if fixed else "#E5E3DA"}" stroke-width="{1 if fixed else 0.5}"/>')
        if share:
            out.append(f'<g class="{cls}"><rect x="{x0}" y="{y}" width="{bw * share:g}" height="20" rx="3" '
                       f'stroke-width="0.5"/></g>')
        text(out, x0 + bw + 10, y + 10, f"{share:.0%}", "tv", "start")
    ly = 196 + 30 * len(kinds) + 16
    for x, cls, label in ((40, "c-teal", "Predicted fixed by the edit"), (280, "box", "Predicted unchanged")):
        out.append(f'<g class="{cls}"><rect x="{x}" y="{ly - 7}" width="14" height="14" rx="3" stroke-width="0.5"/></g>')
        text(out, x + 22, ly, label, "ts", "start")
    write("baseline_kinds.svg", 960, ly + 22, "Where the unedited MiniLM fails",
          "Top: on the three scored labels, MiniLM scores F1 0.674 on the Singapore main eval and 0.984 on Nemotron, "
          "a gap of 0.310, far above the gate's threshold of 0.0087. Bottom: the share of entities right on the main "
          "eval for each kind. NRICs without tax context are 0% right and are the one kind predicted to be fixed. "
          "Later words of given names are 5% right. The other kinds are 62% to 89% right.", out)


data_sets()
example_rows()
label_translation()
tax_id_cases()
harness()
edit_checks()
lock_checks()
set_roles()
baseline_kinds()

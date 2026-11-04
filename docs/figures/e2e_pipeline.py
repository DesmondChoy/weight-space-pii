"""Generate e2e_pipeline.svg, the swimlane figure in docs/m1_report.md."""

from pathlib import Path

W, KEY_H = 960, 214
H = 1350 + KEY_H
LANES = {"A": (125, 315), "B": (330, 520), "C": (535, 725), "D": (745, 940)}
out = []


def cx(lane):
    a, b = LANES[lane]
    return (a + b) / 2


def text(x, y, s, cls, anchor="middle"):
    out.append(f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


def box(x, y, w, h, title, lines=(), cls="box"):
    g = cls if cls != "box" else ""
    out.append(f'<g class="{g}">' if g else "<g>")
    rc = ' class="box"' if cls == "box" else ""
    out.append(f'<rect{rc} x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="8" stroke-width="0.5"/>')
    c = x + w / 2
    text(c, y + 18, title, "th")
    for i, ln in enumerate(lines):
        text(c, y + 38 + 16 * i, ln, "ts")
    out.append("</g>")


def lane_box(lane, y, h, title, lines=(), cls="box"):
    a, b = LANES[lane]
    box(a, y, b - a, h, title, lines, cls)


def arrow(d):
    out.append(f'<path d="{d}" class="arr" marker-end="url(#arrow)"/>')


def row(y0, y1, label, sub):
    out.append(f'<line x1="20" y1="{y0}" x2="{W-20}" y2="{y0}" stroke="#E5E3DA" stroke-width="1"/>')
    text(24, y0 + 20, label, "th", "start")
    text(24, y0 + 38, sub, "ts", "start")


# lane headers
heads = {"A": ("Practice", "c-teal", "editor sees"), "B": ("Main eval", "c-coral", "Singapore, hidden"),
         "C": ("Regression", "c-coral", "Nemotron, hidden"), "D": ("Scoring, comparisons", "c-purple", "fixed protocol")}
for lane, (t, cls, s) in heads.items():
    lane_box(lane, 16, 52, t, [s], cls)

# rows
row(80, 260, "M1", "Environment")
row(260, 480, "M2 to M3", "Profile, sample")
row(480, 570, "M4", "Harness")
row(570, 790, "M5", "Freeze, baseline")
row(790, 960, "M6", "Editor run")
row(960, 1200, "M7", "Evaluate")
row(1200, 1290, "M8", "Write-up")
out.append(f'<line x1="20" y1="1290" x2="{W-20}" y2="1290" stroke="#E5E3DA" stroke-width="1"/>')

# M1
box(125, 94, 395, 52, "openpii Singapore", ["25,230 rows, region SG"])
lane_box("C", 94, 52, "Nemotron-PII", ["200,000 rows, train + test"])
arrow(f"M322 146 L322 174 L{cx('A'):g} 174 L{cx('A'):g} 194")
arrow(f"M322 174 L{cx('B'):g} 174 L{cx('B'):g} 194")
text(330, 160, "openpii's own split", "ts", "start")
arrow(f"M{cx('C'):g} 146 L{cx('C'):g} 194")
text(cx("C") - 8, 172, "author's 90/10", "ts", "end")
lane_box("A", 196, 52, "Train split", ["20,225 rows"])
lane_box("B", 196, 52, "Validation split", ["5,005 rows"])
lane_box("C", 196, 52, "Held-out split", ["20,000 rows, 0 trained on"])

# M2-M3
for l in "ABC":
    arrow(f"M{cx(l):g} 248 L{cx(l):g} 274")
out.append('<rect x="125" y="276" width="600" height="36" rx="4" fill="none" stroke="#888780" stroke-width="1" stroke-dasharray="5 4"/>')
text(425, 294, "Split point: filter, dedupe, seeded practice sample, relabel", "th")
for l in "ABC":
    arrow(f"M{cx(l):g} 312 L{cx(l):g} 336")
lane_box("A", 338, 52, "Practice", ["1,000 rows"], "c-teal")
lane_box("B", 338, 52, "Main eval", ["5,004 rows, all filtered"], "c-coral")
lane_box("C", 338, 52, "Regression", ["18,515 rows, all filtered"], "c-coral")
lane_box("D", 276, 168, "Scoring protocol", [
    "1. Tokenize, max 512", "2. Argmax, first subtoken", "3. BIO repair, word spans", "4. Split names into words",
    "5. Match gold: exact span + type", "Score: strict micro F1,", "precision, recall,", "per-category F1",
], "c-purple")

# M4
arrow(f"M{cx('A'):g} 390 L{cx('A'):g} 492")
lane_box("A", 494, 56, "Dry run", ["2 passes, 10 turns max"])
lane_box("D", 494, 56, "Check: server = scorer", ["same scores, 50 practice rows"], "c-purple")

# M5
for l in "BC":
    arrow(f"M{cx(l):g} 390 L{cx(l):g} 582")
out.append('<rect x="330" y="584" width="395" height="32" rx="4" fill="none" stroke="#888780" stroke-width="1" stroke-dasharray="5 4"/>')
text(527.5, 600, "lock.json: sets, scorer, server, prompt frozen", "ts")
for l in "BC":
    arrow(f"M{cx(l):g} 616 L{cx(l):g} 630")
lane_box("B", 632, 52, "Score base model", ["Singapore F1"], "c-purple")
lane_box("C", 632, 52, "Score base model", ["Nemotron F1"], "c-purple")
lane_box("D", 626, 64, "C1 gate", ["Nemotron F1 − SG F1 > 2 SE,", "else stop"], "c-purple")
arrow("M725 658 L743 658")
arrow(f"M{cx('B'):g} 684 L{cx('B'):g} 702 L{cx('D'):g} 702 L{cx('D'):g} 692")
lane_box("D", 714, 56, "Prediction written", ["fixed or unchanged, per category"], "c-purple")

# M6
arrow(f"M{cx('A'):g} 550 L{cx('A'):g} 802")
lane_box("A", 804, 56, "Editor run", ["50 passes, practice scores"])
arrow(f"M{cx('A'):g} 860 L{cx('A'):g} 882")
lane_box("A", 884, 52, "Edit artifact", ["edit.npz + manifest"])

# M7
arrow(f"M{cx('A'):g} 936 L{cx('A'):g} 948 L{cx('B'):g} 948 L{cx('B'):g} 974")
arrow(f"M{cx('B'):g} 948 L{cx('C'):g} 948 L{cx('C'):g} 974")
lane_box("B", 976, 52, "Score edited model", ["Singapore F1"], "c-purple")
lane_box("C", 976, 52, "Score edited model", ["Nemotron F1"], "c-purple")
lane_box("D", 970, 64, "C3 regression", ["edited − base on Nemotron,", "paired ± SE, near zero"], "c-purple")
arrow("M725 1002 L743 1002")
lane_box("D", 1048, 64, "C2 Singapore gain", ["edited − base on main eval,", "paired ± SE"], "c-purple")
arrow(f"M{cx('B')+20:g} 1028 L{cx('B')+20:g} 1080 L743 1080")
lane_box("D", 1126, 64, "C4 mechanism metric", ["targeted failure share,", "base vs edited"], "c-purple")
arrow(f"M{cx('B')-20:g} 1028 L{cx('B')-20:g} 1158 L743 1158")

# M8
box(125, 1214, 600, 56, "Write-up and demo", ["Main table with C1 to C5, side-by-side claims letter"])
lane_box("D", 1210, 64, "C5 prediction check", ["per-category change", "vs the M5 prediction"], "c-purple")
arrow("M745 1242 L727 1242")

# visibility boundary
out.append('<line x1="322.5" y1="320" x2="322.5" y2="1196" stroke="#888780" stroke-width="1" stroke-dasharray="4 4"/>')

# legend
ly = 1318
items = [("c-teal", "Editor can see"), ("c-coral", "Hidden from editor"), ("c-purple", "Scoring and comparisons")]
x = 125
for cls, label in items:
    out.append(f'<rect class="{cls}" x="{x}" y="{ly-7}" width="14" height="14" rx="3" stroke-width="0.5"/>')
    text(x + 22, ly, label, "ts", "start")
    x += 200
out.append(f'<line x1="{x}" y1="{ly}" x2="{x+30}" y2="{ly}" stroke="#888780" stroke-width="1" stroke-dasharray="4 4"/>')
text(x + 40, ly, "Visibility boundary", "ts", "start")


MILESTONES = [
    ("M1 Environment", "Pin the Python env, download model and data, sign in the editor"),
    ("M2 Profile", "Profile the Singapore rows, propose sample sizes and scored labels"),
    ("M3 Sample and metric", "Draw the seeded practice sample, relabel, write and test the scorer"),
    ("M4 Harness", "Build the driver, kernel and forward-pass server; 2-pass dry run"),
    ("M5 Freeze and baseline", "Lock eval sets and code, score the base model, write the prediction"),
    ("M6 Editor run", "Opus 5.5 edits the weights directly, with 50 forward passes"),
    ("M7 Evaluate", "Rebuild the edited weights, score them on the locked sets"),
    ("M8 Write-up", "Main table, prediction check, side-by-side demo"),
]
main = out
out = []
text(24, 26, "Milestones. Each one ends with a stop for review before the next starts.", "th", "start")
for i, (name, desc) in enumerate(MILESTONES):
    x, y = (24 if i < 4 else 490), 56 + 42 * (i % 4)
    text(x, y, name, "th", "start")
    text(x, y + 18, desc, "ts", "start")
key_svg = out
out = main

head = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">
<title>End-to-end weight-space PII experiment</title>
<desc>Swimlane diagram. Rows are milestones M1 to M8. Columns are the practice set the editor sees, the hidden Singapore main eval, the hidden Nemotron regression set, and a lane for scoring and comparisons. Data is downloaded and split at M1, sampled and relabelled at M3 where the scoring protocol is fixed, checked at M4, frozen and scored for the base model at M5 with the C1 gate, edited at M6, scored for the edited model at M7 with comparisons C2 to C4, and written up at M8 with comparison C5.</desc>
<style>
text {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
.th {{ font-size: 14px; font-weight: 500; fill: #2C2C2A; }}
.ts {{ font-size: 12px; font-weight: 400; fill: #5F5E5A; }}
.box {{ fill: #F1EFE8; stroke: #888780; }}
.arr {{ stroke: #5F5E5A; stroke-width: 1.5; fill: none; }}
.c-teal rect, rect.c-teal {{ fill: #E1F5EE; stroke: #0F6E56; }}
.c-teal .th {{ fill: #085041; }} .c-teal .ts {{ fill: #0F6E56; }}
.c-coral rect, rect.c-coral {{ fill: #FAECE7; stroke: #993C1D; }}
.c-coral .th {{ fill: #712B13; }} .c-coral .ts {{ fill: #993C1D; }}
.c-purple rect, rect.c-purple {{ fill: #EEEDFE; stroke: #534AB7; }}
.c-purple .th {{ fill: #3C3489; }} .c-purple .ts {{ fill: #534AB7; }}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#5F5E5A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
'''
body = "\n".join(key_svg) + f'\n<g transform="translate(0,{KEY_H})">\n' + "\n".join(out) + "\n</g>"
(Path(__file__).with_suffix(".svg")).write_text(head + body + "\n</svg>\n")

"""Generate scoring_rules.svg: where the M3 scoring rules live, and what they do to two practice rows."""

from pathlib import Path

W, H = 960, 1160
CW = 7.8
out = []


def text(x, y, s, cls, anchor="start"):
    out.append(f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


def lines(x, y, rows, cls="ts", step=16):
    for i, s in enumerate(rows):
        text(x, y + step * i, s, cls)


def box(x, y, w, h, title, rows=(), cls="box"):
    out.append(f'<g class="{cls}"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="8" stroke-width="0.5"/>')
    text(x + w / 2, y + 18, title, "th", "middle")
    for i, s in enumerate(rows):
        text(x + w / 2, y + 38 + 16 * i, s, "ts", "middle")
    out.append("</g>")


def arrow(d):
    out.append(f'<path d="{d}" class="arr" marker-end="url(#arrow)"/>')


def badge(x, y, s):
    out.append(f'<circle cx="{x:g}" cy="{y:g}" r="10" class="badge"/>')
    text(x, y, s, "bt", "middle")


def snippet(x0, y, s):
    out.append(f'<text class="mono" x="{x0:g}" y="{y:g}" dominant-baseline="central" '
               f'textLength="{len(s) * CW:g}" lengthAdjust="spacingAndGlyphs" xml:space="preserve">{s}</text>')


def mark(x0, y, s, word, cls, label="", dashed=False):
    i = s.index(word)
    x, w = x0 + i * CW - 2, len(word) * CW + 4
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    out.append(f'<rect class="{cls}" x="{x:g}" y="{y - 11:g}" width="{w:g}" height="22" rx="4" stroke-width="1"{dash}/>')
    if label:
        text(x + w / 2, y + 22, label, f"tl {cls}", "middle")


def row_label(y, title, sub):
    text(36, y - 6, title, "th")
    text(36, y + 11, sub, "ts")


def panel(y, h, title, sub):
    out.append(f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="10" fill="#FAFAF7" stroke="#E5E3DA"/>')
    text(36, y + 22, title, "th")
    text(36, y + 40, sub, "ts")


text(20, 24, "How Opus 5.5's grades are scored, and what the M4 prompt must tell it", "h1")
text(20, 48, "M3 added three scoring rules. Every grade Opus 5.5 gets applies them, so the task prompt has to state them.", "ts")

# where the rules live
text(20, 84, "Where the rules live", "th")
text(400, 84, "Opus 5.5 can see", "tc")
text(780, 84, "Hidden from Opus 5.5", "tr")
box(20, 104, 270, 66, "§6 task prompt, filled in at M4", ["Brief says: exact span and type", "M4 adds: rules 1, 2 and 3"], "c-teal")
box(20, 190, 270, 66, "Practice rows, sg_train.jsonl", ["id, text, spans", "M3 adds: ID numbers the scoring skips"], "c-teal")
box(330, 104, 150, 152, "Opus 5.5", ["reads the prompt,", "the rows and", "every grade"], "c-teal")
box(560, 104, 180, 56, "Forward-pass server", ["built in M4"])
box(560, 196, 180, 108, "wsp.score.grade", ["1. Split names into words", "2. tax_id on an NRIC counts", "3. Skip some ID numbers", "then exact span and type"], "c-purple")
box(790, 196, 150, 48, "M5 baseline", ["main eval, Nemotron"], "c-coral")
box(790, 256, 150, 48, "M7 evaluation", ["edited against base"], "c-coral")
arrow("M290 137 L328 137")
arrow("M290 223 L328 223")
arrow("M480 122 L558 122")
text(519, 112, "request", "tl", "middle")
arrow("M560 144 L482 144")
text(521, 156, "grades", "tl", "middle")
arrow("M650 160 L650 194")
text(658, 177, "calls", "tl")
arrow("M790 220 L742 220")
arrow("M790 280 L742 280")
out.append('<line x1="765" y1="96" x2="765" y2="312" stroke="#888780" stroke-width="1" stroke-dasharray="4 4"/>')
badge(290, 104, "A")
badge(290, 190, "B")
badge(632, 177, "C")

lines(60, 336, [
    "The prompt must state rules 1 to 3. Without them, the grades Opus 5.5 gets contradict the rule it was given.",
    "The prompt must say that each practice row lists the ID numbers the scoring skips, and what that means.",
    "The server must call wsp.score.grade, not its own copy, so practice grades follow the same rules as M5 and M7.",
], "ts", 24)
for i, s in enumerate("ABC"):
    badge(40, 336 + 24 * i, s)

# panel 1: names
S1 = "full name (Dragos-Costin Keyfo Alhadeff Hachen), birthdate"
X1, P1 = 190, 420
panel(P1, 250, "Rule 1. Each name is scored one word at a time", "Practice row sg-24731778, which Opus 5.5 can read")
y = P1 + 82
row_label(y, "In the file", "openpii's spans")
mark(X1, y, S1, "Dragos-Costin Keyfo", "first", "first_name, one span")
mark(X1, y, S1, "Alhadeff Hachen", "last", "last_name, one span")
snippet(X1, y, S1)
y += 60
row_label(y, "MiniLM's output", "one entity per word")
for word, cls, lab in [("Dragos-Costin", "first", "first_name"), ("Keyfo", "last", "last_name"),
                       ("Alhadeff", "last", "last_name"), ("Hachen", "last", "last_name")]:
    mark(X1, y, S1, word, cls, lab)
snippet(X1, y, S1)
y += 60
row_label(y, "Scoring code", "gold split into words")
for word, cls, ok in [("Dragos-Costin", "first", True), ("Keyfo", "first", False),
                      ("Alhadeff", "last", True), ("Hachen", "last", True)]:
    mark(X1, y, S1, word, cls)
    i = S1.index(word)
    text(X1 + (i + len(word) / 2) * CW, y + 22, "✓" if ok else "✗", "ok" if ok else "bad", "middle")
snippet(X1, y, S1)

VX = 680
text(VX, P1 + 74, "3 of 4 right", "big ok")
text(VX, P1 + 94, "with rule 1, which the server reports", "ts")
text(VX, P1 + 124, "0 of 2 right", "big bad")
text(VX, P1 + 144, "read literally: no whole name matches", "ts")
lines(VX, P1 + 176, [
    "The one real error is Keyfo, a given name", "typed as last_name. Without rule 1,",
    "Opus 5.5 would chase whole-name matches,", "which the score doesn't reward.",
])

# panel 2: IDs
S2 = "5. IDs: National ID XCN5487R2W, Tax ID T3215794T"
X2, P2 = 190, 690
panel(P2, 300, "Rules 2 and 3. Two ID numbers in one row", "Practice row sg-24844004. The scoring skips its ID card number")
y = P2 + 82
row_label(y, "In the file", "spans and a skipped ID")
mark(X2, y, S2, "XCN5487R2W", "skip", "skipped: IDCARDNUM", dashed=True)
mark(X2, y, S2, "T3215794T", "nric", "national_id, an NRIC")
snippet(X2, y, S2)
y += 60
row_label(y, "MiniLM's output", "follows the words")
mark(X2, y, S2, "XCN5487R2W", "nric", "national_id")
mark(X2, y, S2, "T3215794T", "nric", "tax_id")
snippet(X2, y, S2)
y += 60
row_label(y, "Scoring code", "after rules 2 and 3")
mark(X2, y, S2, "XCN5487R2W", "skip", "not counted, rule 3", dashed=True)
mark(X2, y, S2, "T3215794T", "nric", "✓ national_id, rule 2")
snippet(X2, y, S2)

text(VX, P2 + 74, "1 right, 0 false positives", "big ok")
text(VX, P2 + 94, "with rules 2 and 3", "ts")
text(VX, P2 + 124, "0 right, 1 false positive", "big bad")
text(VX, P2 + 144, "read literally", "ts")
lines(VX, P2 + 176, [
    "Rule 2. The text says Tax ID, and MiniLM", "follows it. On practice, 134 of 159 NRICs",
    "after \"tax\" are right, and 0 of the other 61.",
])
lines(VX, P2 + 232, [
    "Rule 3. M2 dropped ID card, passport and", "licence numbers. A prediction on one",
    "counts neither way. national_id false", "positives fall from 65 to 5.",
])

# chart
C = 1020
text(20, C, "What each rule changes on the whole practice set", "th")
text(20, C + 18, "Base model on all 1,000 practice rows. The predictions are the same in every bar. Only the scoring rules change.", "ts")
bars = [
    ("All three rules (the server)", 0.682, "c-purple", ""),
    ("Without rule 1 (names)", 0.469, "c-gray", "names: first 0.59 → 0.45, last 0.75 → 0.44"),
    ("Without rule 2 (tax_id)", 0.646, "c-gray", "national_id 0.75 → 0.00"),
    ("Without rule 3 (skipped IDs)", 0.674, "c-gray", "national_id 0.75 → 0.64"),
    ("None of them (brief's wording)", 0.416, "c-coral", ""),
]
for i, (lab, v, cls, note) in enumerate(bars):
    by = C + 44 + 16 * i
    text(36, by, lab, "ts")
    out.append(f'<rect class="{cls}" x="250" y="{by - 6}" width="{v * 520:g}" height="12" rx="2" stroke-width="0.5"/>')
    text(256 + v * 520, by, f"{v:.3f}", "tv")
    if note:
        text(650, by, note, "tl")

head = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">
<title>Scoring rules Opus 5.5 must be told about</title>
<desc>Top: Opus 5.5 reads the task prompt, the practice rows and the grades from the forward-pass server. The server calls wsp.score.grade, which applies three rules and also scores the hidden M5 and M7 sets. Middle: two practice rows. In the first, a two-word given name scores 3 of 4 words right when each word of a name is scored separately, and 0 of 2 read literally. In the second, a tax_id prediction on an NRIC counts as right and a national_id prediction on an ID card number the scoring skips is not counted. Bottom: practice F1 is 0.682 with all three rules and 0.416 without them.</desc>
<style>
text {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
.h1 {{ font-size: 17px; font-weight: 600; fill: #2C2C2A; }}
.th {{ font-size: 14px; font-weight: 500; fill: #2C2C2A; }}
.ts {{ font-size: 12px; fill: #5F5E5A; }}
.tl {{ font-size: 11px; fill: #5F5E5A; }}
.tv {{ font-size: 12px; font-weight: 500; fill: #2C2C2A; }}
.tc {{ font-size: 12px; fill: #0F6E56; text-anchor: middle; }}
.tr {{ font-size: 12px; fill: #993C1D; text-anchor: start; }}
.big {{ font-size: 15px; font-weight: 600; }}
.ok {{ fill: #3B6D11; font-size: 13px; font-weight: 600; }} .bad {{ fill: #A32D2D; font-size: 13px; font-weight: 600; }}
.big.ok, .big.bad {{ font-size: 15px; }}
.mono {{ font-family: Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 13px; fill: #2C2C2A; }}
.bt {{ font-size: 11px; font-weight: 700; fill: #FFFFFF; }}
.badge {{ fill: #2C2C2A; }}
.box rect {{ fill: #F1EFE8; stroke: #888780; }}
.arr {{ stroke: #5F5E5A; stroke-width: 1.5; fill: none; }}
.c-teal rect {{ fill: #E1F5EE; stroke: #0F6E56; }} .c-teal .th {{ fill: #085041; }} .c-teal .ts {{ fill: #0F6E56; }}
.c-coral rect, rect.c-coral {{ fill: #FAECE7; stroke: #993C1D; }} .c-coral .th {{ fill: #712B13; }} .c-coral .ts {{ fill: #993C1D; }}
.c-purple rect, rect.c-purple {{ fill: #EEEDFE; stroke: #534AB7; }} .c-purple .th {{ fill: #3C3489; }} .c-purple .ts {{ fill: #534AB7; }}
rect.c-gray {{ fill: #D3D1C7; stroke: #888780; }}
rect.first {{ fill: #E6F1FB; stroke: #185FA5; }} text.first {{ fill: #0C447C; }}
rect.last {{ fill: #FAEEDA; stroke: #854F0B; }} text.last {{ fill: #633806; }}
rect.nric {{ fill: #FBEAF0; stroke: #993556; }} text.nric {{ fill: #72243E; }}
rect.skip {{ fill: #F1EFE8; stroke: #888780; }} text.skip {{ fill: #5F5E5A; }}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#5F5E5A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
'''
Path(__file__).with_suffix(".svg").write_text(head + "\n".join(out) + "\n</svg>\n")

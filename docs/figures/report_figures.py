"""Generate the figures in the M2 and M3 reports: data_sets, example_rows, label_translation, and tax_id_cases."""

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


data_sets()
example_rows()
label_translation()
tax_id_cases()

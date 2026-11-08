"""Build the M7 review pack: every item Desmond reviewed to approve M7, with how Opus 5.5 reached each result.

Reads runs/p50/eval/, which scripts/evaluate.py writes, and the sets. Writes one HTML page.

Usage: uv run python -I scripts/m7_review_page.py results/m7_review.html
"""

import json
import random
import sys
from html import escape
from pathlib import Path

from wsp.data import ROOT, SETS, read_jsonl

OUT = Path(sys.argv[1])
ev = json.loads((ROOT / "runs/p50/eval/evaluation.json").read_text())
rows_all = [json.loads(l) for l in (ROOT / "runs/p50/eval/changed_rows.jsonl").open()]
S = ev["scores"]
sg, nem, gate = S["singapore"]["bootstrap"], S["nemotron_regression"]["bootstrap"], S["nemotron_gate"]["bootstrap"]
kinds = ev["kinds"]["verdicts"]
titles = ev["mechanism"]["titles"]
nric = ev["mechanism"]["nric"]["national_id/no tax context"]

KIND_NAMES = {"national_id/no tax context": 'NRIC without "tax" just before it',
              "national_id/tax context": 'NRIC with "tax" just before it',
              "first_name/one word": "Given name, one word",
              "first_name/multi-word, first word": "Given name, several words, first word",
              "first_name/multi-word, later word": "Given name, several words, later words",
              "last_name/one word": "Surname, one word",
              "last_name/multi-word, first word": "Surname, several words, first word",
              "last_name/multi-word, later word": "Surname, several words, later words"}


def pct(x, d=0):
    from decimal import ROUND_HALF_UP, Decimal
    return f"{(Decimal(str(x)) * 100).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP)}%"


def kind_rows():
    out = []
    for k, name in KIND_NAMES.items():
        v = kinds[k]
        pred = "fixed" if k == "national_id/no tax context" else "unchanged"
        match = "match" if v["verdict"] == pred else "miss"
        out.append(f"""<tr><th scope="row">{escape(name)} <span class="n">{v['entities']:,}</span></th>
<td class="bars"><div class="bar base" style="--w:{v['base_share']}"><span>{pct(v['base_share'])}</span></div>
<div class="bar ed" style="--w:{v['edited_share']}"><span>{pct(v['edited_share'])}</span></div></td>
<td class="num">{v['change_pp']:+.2f}</td><td>{pred}</td><td><span class="pill {match}">{v['verdict']}</span></td></tr>""")
    return "\n".join(out)


def title_rows():
    out = []
    for g, label in (("targeted", "Targeted"), ("control", "Control"), ("untargeted", "Untargeted")):
        for w, v in titles["words"][g].items():
            if not v["n"]:
                continue
            out.append(f"<tr><td>{label}</td><td><code>{w}</code></td><td class='num'>{v['n']}</td>"
                       f"<td class='num'>{v['base_hit']} <span class='n'>{pct(v['base_share'], 1)}</span></td>"
                       f"<td class='num'>{v['edited_hit']} <span class='n'>{pct(v['edited_share'], 1)}</span></td></tr>")
    return "\n".join(out)


def regression_rows():
    r = ev["nemotron_regression_by_label"]
    shown = r[:6] + r[-3:]
    out = []
    for x in shown:
        cls = "neg" if x["change"] < 0 else "pos"
        w = min(abs(x["change"]) / 0.05, 1)
        out.append(f"<tr><td><code>{x['label']}</code></td><td class='num'>{x['base_f1']:.3f}</td>"
                   f"<td class='num'>{x['edited_f1']:.3f}</td><td class='delta'><div class='dbar {cls}' "
                   f"style='--w:{w}'></div><span class='num'>{x['change']:+.4f}</span></td></tr>")
    return "\n".join(out)


def check_rows():
    return "\n".join(f"<li class='{'ok' if c['passed'] else 'fail'}'><span class='mark'>{'Pass' if c['passed'] else 'Fail'}</span>"
                     f"{escape(c['name'])}</li>" for c in ev["checks"])


def highlight(text, spans, start, end):
    marks = sorted(spans)
    out, pos = [], start
    for a, b, cls in marks:
        if a < pos or b > end:
            continue
        out.append(escape(text[pos:a]))
        out.append(f"<mark class='{cls}'>{escape(text[a:b])}</mark>")
        pos = b
    out.append(escape(text[pos:end]))
    return "".join(out).replace("\n", " <span class='br'>⏎</span> ")


def gold_at(d, a, b):
    ex = [g.label for g in d.spans if (g.start, g.end) == (a, b)]
    if ex:
        return ex[0]
    ov = [g.label for g in d.spans if g.start < b and a < g.end]
    return f"overlaps {ov[0]}" if ov else "none"


def changed_cards():
    rng = random.Random(20261106)
    groups = {}
    n = 0
    for set_name in ("sg_eval", "nemotron_eval"):
        docs = read_jsonl(SETS / f"{set_name}.jsonl")
        for status in ("fixed", "broken"):
            pool = [r for r in rows_all if r["set"] == set_name and r["status"] == status]
            cards = []
            for r in sorted(rng.sample(pool, 5), key=lambda r: r["position"]):
                n += 1
                d = docs[r["position"]]
                assert d.id == r["id"]
                base = {(s[0], s[1]): s[2] for s in r["base_only_spans"]}
                edited = {(s[0], s[1]): s[2] for s in r["edited_only_spans"]}
                keys = sorted(base.keys() | edited.keys())
                lo = max(0, min(a for a, _ in keys) - 160)
                hi = min(len(d.text), max(b for _, b in keys) + 160)
                spans = []
                for a, b in keys:
                    if any(a < y and x < b for x, y, _ in spans):
                        continue
                    spans.append((a, b, "chg"))
                quote = ("…" if lo else "") + highlight(d.text, spans, lo, hi) + ("…" if hi < len(d.text) else "")
                trs = "".join(f"<tr><td><code>{escape(d.text[a:b])}</code></td><td>{escape(gold_at(d, a, b))}</td>"
                              f"<td>{escape(base.get((a, b), 'nothing'))}</td><td>{escape(edited.get((a, b), 'nothing'))}</td></tr>"
                              for a, b in keys)
                be, ee = r["base"]["fp"] + r["base"]["fn"], r["edited"]["fp"] + r["edited"]["fn"]
                cards.append(f"""<article class="row {status}"><header><span class="idx">{n}</span>
<code class="rid">{escape(d.id)}</code><span class="err">errors {be} → {ee}</span></header>
<blockquote>{quote}</blockquote>
<div class="tbl"><table class="tags"><thead><tr><th>Text</th><th>Gold label</th><th>Unedited</th><th>Edited</th></tr></thead>
<tbody>{trs}</tbody></table></div></article>""")
            groups[(set_name, status)] = "\n".join(cards)
    return groups


trans = ev["nemotron_transitions"][0]


def how(title, items, source):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return (f'<div class="how"><h3>{title}</h3><ol>{lis}</ol>'
            f'<p class="src">Source: {source}</p></div>')


def facts(*xs):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"


NOTES = "<code>runs/p50/work_notes/notes.md</code>"
ANSWER = "<code>runs/p50/final_answer.md</code>"

HOW_MAIN = how("How Opus 5.5 got this result", [
    "<b>It kept half the practice set for testing.</b> The practice set has 1,000 Singapore rows. Opus 5.5 built its edits "
    "on 500 of them, the dev half, and tested them on the other 500, the held-back half.",
    "<b>It found two failures before it changed any weights.</b>"
    + facts("MiniLM tagged Singapore titles, such as Madame, Miss and Mstr, as given names. It never did this to Mr or Mrs.",
            "MiniLM chose an NRIC's ID type from a nearby word such as \"tax\". With no such word, it got every NRIC wrong."),
    "<b>It fixed each failure by rewiring a few unused neurons.</b> MiniLM has neurons that almost never fire. Opus 5.5 "
    "set their weights by hand, so that they fire on one thing, such as an NRIC. When they fire, they push MiniLM toward the "
    "answer it already gives when it gets that case right. Opus 5.5 calculated the weights from MiniLM's own activations. "
    "It trained nothing.",
    "<b>It tried 19 edits and kept two repairs.</b> A pass is one run of MiniLM that Opus 5.5 asked for, and it had 50. "
    "It also built a repair for hyphenated names, then dropped it because its gain was within noise.",
    f"<b>Its own estimate was a little high.</b> It measured +0.065 on the held-back half. But it had also used that half "
    f"to choose which repairs to keep. The main eval, which it never saw, gives {sg['diff']:+.3f}.",
], f"{NOTES}, {ANSWER}, and each pass's <code>forward/&lt;k&gt;/edit_manifest.json</code>.")

HOW_KINDS = how("What happened in each category", [
    "<b>NRICs without \"tax\" were fixed.</b> MiniLM could already tell that a code looked like an NRIC. It just never used "
    "that to choose the ID type. Opus 5.5's neurons in layer 4 fire on the NRIC's shape and push MiniLM to "
    "<code>national_id</code>.",
    f"<b>The fix beat the prediction's limit.</b> The prediction assumed that MiniLM would keep missing the 118 NRICs it never "
    f"tagged, so at most 205 of 323 could become right. {nric['edited_correct']} did. So the edit also made MiniLM tag at "
    f"least {nric['edited_correct'] - 205} NRICs that it used to skip.",
    "<b>NRICs after \"tax\" were almost fixed.</b> These already counted as right when MiniLM called them "
    "<code>tax_id</code>. The same neurons fire here and turned some wrong answers into <code>national_id</code>. The share "
    "rose 9.99 points, just under the 10-point line.",
    "<b>The name categories did not change.</b> These categories count how many real names MiniLM tags. The title repair "
    "stops MiniLM from tagging the title, and a title is never a real name. So the repair shows up as fewer false positives, "
    "not in these categories.",
    "<b>Opus 5.5 tried to make MiniLM find more names, and failed.</b> It tried a repair for hyphenated names and a repair "
    "for names used as chat speaker labels. The first gained too little. The second created about as many false positives as "
    "it fixed, because MiniLM's last layer cannot tell rare speaker names from other words. This fits the paper. An edit can "
    "remove an obstacle, but it cannot teach MiniLM a name it does not know.",
], f"{NOTES}, the sections on the NRIC failure, the speaker labels, and the scores of each repair alone.")

HOW_MECH = how("How Opus 5.5 built and tested each repair", [
    "<b>Mr and Mrs carry a title signal, and Singapore titles do not.</b> The signal tells MiniLM's last layer to tag "
    "nothing. Singapore titles lack it partly because the tokenizer splits them into pieces that look like names, such as "
    "<code>mayor ##ess</code> and <code>ms ##tr</code>.",
    "<b>The first title repair came too late in MiniLM.</b> Opus 5.5 first added the signal in layer 2. MiniLM stopped "
    "tagging the titles, but it still read the name after \"Master\" as a company. That name looks back at the title in "
    "layers 0 to 2, before the layer-2 repair runs. Moving the repair to layer 1 fixed most of these. This is why names right "
    "after a title improved more than other names."
    + facts("Names after \"Master\" right on the dev half: 69 of 95 with the repair in layer 2, 81 in layer 1, and 85 with "
            "the signal doubled."),
    "<b>It checked that MiniLM really uses the signal.</b> It removed the signal from \"Mrs\". MiniLM then tagged Mrs as a "
    "name in all 3 test sentences. Opus 5.5 then undid this test.",
    "<b>It caught a false trigger late in the run.</b> In long US texts, the title repair fired on \"Mary\". Opus 5.5 "
    "recalculated it with 258 US first names as examples of what not to fire on. Afterwards, no name in 281 test cases lost "
    "its tag.",
    "<b>It chose how hard the NRIC repair pushes.</b> A stronger push found more NRICs on the dev half. But at 1.5 times the "
    "base strength it gained nothing on the held-back half and mislabelled more look-alike codes. So Opus 5.5 kept 1 times."
    + facts("Share of dev NRICs found at strength 0, 0.5, 1 and 1.5: 61%, 66%, 96% and 99%."),
    "<b>It knew about one gap.</b> \"Mdm\" never appears in the practice set, so the repair does not cover it. The main eval "
    "has no \"Mdm\" either, so the table cannot show it.",
], f"{NOTES} and {ANSWER}.")

HOW_NEM = how("How Opus 5.5 protected other text, and what it missed", [
    "<b>It tested about 300 US-style texts that it wrote.</b> These stand in for MiniLM's usual data, which Opus 5.5 never saw.",
    "<b>Those tests found side effects, and it fixed them.</b> Before the edit, the rewired neurons gave a tiny output on every "
    "word. Removing that output tipped a few close calls. For example, a passport number changed from "
    "<code>national_id</code> to <code>pin</code>. Opus 5.5 added the old average output back into the layer's output bias. "
    "So the edit changes all 384 numbers in the output bias of each edited layer.",
    "<b>It tested codes that look like NRICs.</b>"
    + facts("A letter and 7 or 8 digits kept its label 92% of the time.",
            "Licence and student IDs with their own cue word kept their label 24 times out of 24.",
            "Random codes, two-letter codes, and codes that start with a digit always kept their label."),
    f"<b>It never tested longer codes, and that is where Nemotron broke.</b> Nemotron's biometric IDs, such as "
    f"<code>F98324157683</code>, are a letter and 11 digits. {trans['shapes']['A99999999999']} of the 57 new false "
    "<code>national_id</code> tags have that shape. Opus 5.5's notes say that the NRIC repair sees the first letter clearly "
    "but barely sees how many digits follow. That would explain the failure, but nobody has tested it. Option 2 in item 7 "
    "tests it.",
], f"{NOTES}, the sections on the NRIC repair and on safety for standard text, and {ANSWER}.")

HOW_ROWS = how("What the sample shows", [
    "<b>All five fixed Singapore rows come from the title repair.</b>"
    + facts("In rows 1, 3, 4 and 5, the title Mstr, Madame or Miss no longer gets a name tag.",
            "In row 2, MiniLM now finds the name after \"Master\".",
            "Rows 3 and 4 also change an NRIC from <code>tax_id</code> to <code>national_id</code>. Both count as right.",
            "Only <code>name</code> and <code>national_id</code> are scored on Singapore text, so the new label on the card "
            "number in row 4 does not count."),
    "<b>Two broken Singapore rows point to known weak spots.</b>"
    + facts("In row 6, a 16-digit card number becomes <code>national_id</code>. That is the length problem from Nemotron.",
            "In row 9, a surname right after \"Madame\" loses its name tag. Here the title repair may reach past the title.",
            "In row 7, MiniLM now finds the given name after \"Master\". But it splits the surname M'halla into two name "
            "tags, which counts as two false positives.",
            "In rows 8 and 10, a surname loses its name tag. Row 10 is a chat speaker label, which Opus 5.5 tried and failed "
            "to fix."),
    "<b>No fixed Nemotron row involves a title or an NRIC.</b> They are small changes to the label or the span of an "
    "occupation, a company, a phone number, and dates.",
    "<b>One broken Nemotron row is the ID problem. The rest are scattered.</b>"
    + facts("In row 16, MiniLM now tags the prefix of the biometric ID <code>BIO-7834561290</code> as "
            "<code>national_id</code>.",
            "In rows 17 to 20, MiniLM splits an email in two and tags \"123ms\" as a time, \"The\" as a cookie, and "
            "\"2028\" as a date."),
    "<b>The scattered changes are probably close calls that the edit tipped.</b> Opus 5.5 found and fixed this kind of side "
    "effect during the run, but only on the texts it tried. Nobody has tested this explanation.",
], "the rows below and <code>runs/p50/eval/changed_rows.jsonl</code>.")

HOW_CHECKS = how("What these checks prove", [
    "<b>The scored edit is the one Opus 5.5 submitted.</b> The script rebuilt the edit from <code>edit.npz</code> and the "
    "original weights. All 12 changed tensors match the hashes that the run recorded.",
    "<b>Every edit from the run rebuilds.</b> The run used 19 different edits, and each one matches its record. The submitted "
    "edit is the one first built for pass 39.",
    "<b>The run used the frozen code.</b> Between the freeze and the run, only three report and script files changed. None of "
    "them is harness code.",
    "<b>This scoring matches what Opus 5.5 saw.</b> On the 500 rows of pass 50, this script and the harness give the same "
    "predictions for both models.",
    "<b>Before and after are compared fairly.</b> The base predictions, the row order, and the software versions match the "
    "M5 baseline.",
], "<code>scripts/evaluate.py</code> and the <code>checks</code> list in <code>runs/p50/eval/evaluation.json</code>.")


cards = changed_cards()
cnt = ev["changed_row_counts"]
by_title = ev["names_by_title"]

html = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M7 Review Pack</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
/* Layout: one reading column of review items in the order the brief lists them, each a numbered item you tick off. */
:root {{
  --bg: #f6f7f5; --surface: #ffffff; --fg: #1d2421; --muted: #5d6a64; --line: #dde3df;
  --accent: #0f6e56; --accent-soft: #e1f5ee; --base: #c9cec9; --bad: #a8402a; --bad-soft: #fbe9e3;
  --mark: #fff2b8;
  --display: "IBM Plex Serif", Georgia, serif; --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121614; --surface: #1a201d; --fg: #e4ebe7; --muted: #9aa8a1; --line: #2c3531;
  --accent: #5fcfa8; --accent-soft: #173a2f; --base: #4a524e; --bad: #f08a6f; --bad-soft: #3d201a;
  --mark: #574a12; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121614; --surface: #1a201d; --fg: #e4ebe7; --muted: #9aa8a1; --line: #2c3531;
  --accent: #5fcfa8; --accent-soft: #173a2f; --base: #4a524e; --bad: #f08a6f; --bad-soft: #3d201a;
  --mark: #574a12; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }}
main {{ max-width: 980px; margin: 0 auto; display: grid; gap: 40px; }}
h1, h2 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }}
h1 {{ font-size: 2rem; line-height: 1.2; }}
h2 {{ font-size: 1.35rem; display: flex; gap: 12px; align-items: baseline; }}
h3 {{ font-size: 1rem; margin: 0; }}
p {{ margin: 0; max-width: 68ch; }}
code {{ font-family: var(--mono); font-size: .86em; }}
.eyebrow {{ font-family: var(--mono); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.intro {{ display: grid; gap: 12px; }}
.num, td.num {{ font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: .9em; }}
.n {{ color: var(--muted); font-family: var(--mono); font-size: .8em; font-weight: 400; }}
section {{ display: grid; gap: 16px; }}
.check {{ display: flex; align-items: center; gap: 8px; margin-left: auto; font-family: var(--body); font-size: .85rem; font-weight: 500; color: var(--muted); cursor: pointer; }}
.check input {{ width: 18px; height: 18px; accent-color: var(--accent); }}
.step {{ font-family: var(--mono); font-size: .85rem; color: var(--accent); }}
.headline {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; display: grid; gap: 4px; }}
.stat b {{ font-family: var(--mono); font-size: 1.35rem; font-weight: 500; }}
.stat .delta-good {{ color: var(--accent); }} .stat .delta-bad {{ color: var(--bad); }}
.stat small {{ color: var(--muted); }}
.tbl {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: var(--surface); }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
thead th {{ font-size: .78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
tbody th {{ font-weight: 500; }}
td.bars {{ width: 42%; min-width: 180px; }}
.bar {{ position: relative; height: 15px; margin: 2px 0; background: transparent; }}
.bar::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: calc(var(--w) * 85%); border-radius: 3px; }}
.bar.base::before {{ background: var(--base); }} .bar.ed::before {{ background: var(--accent); }}
.bar span {{ position: absolute; left: calc(var(--w) * 85% + 6px); top: -2px; font-family: var(--mono); font-size: .75rem; color: var(--muted); }}
.pill {{ display: inline-block; padding: 1px 8px; border-radius: 99px; font-size: .8rem; font-weight: 500; }}
.pill.match {{ background: var(--accent-soft); color: var(--accent); }} .pill.miss {{ background: var(--bad-soft); color: var(--bad); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: .85rem; color: var(--muted); }}
.legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.callout {{ border-left: 3px solid var(--bad); background: var(--bad-soft); padding: 12px 16px; border-radius: 0 6px 6px 0; display: grid; gap: 6px; }}
.callout.good {{ border-color: var(--accent); background: var(--accent-soft); }}
.delta {{ display: flex; align-items: center; gap: 8px; min-width: 160px; }}
.dbar {{ height: 12px; width: calc(var(--w) * 110px); border-radius: 2px; }}
.dbar.neg {{ background: var(--bad); }} .dbar.pos {{ background: var(--accent); }}
.checks {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }}
.checks li {{ display: flex; gap: 10px; align-items: baseline; }}
.mark {{ font-family: var(--mono); font-size: .75rem; padding: 0 6px; border-radius: 3px; background: var(--accent-soft); color: var(--accent); flex: none; }}
.fail .mark {{ background: var(--bad-soft); color: var(--bad); }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.tabs button {{ font: inherit; font-size: .88rem; padding: 6px 12px; border: 1px solid var(--line); background: var(--surface); color: var(--fg); border-radius: 99px; cursor: pointer; }}
.tabs button[aria-selected="true"] {{ background: var(--fg); color: var(--bg); border-color: var(--fg); }}
.tabs button:focus-visible, .check input:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.panel {{ display: grid; gap: 14px; }}
.row {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; display: grid; gap: 10px; min-width: 0; }}
.row.fixed {{ border-top: 3px solid var(--accent); }} .row.broken {{ border-top: 3px solid var(--bad); }}
.row header {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }}
.idx {{ font-family: var(--mono); color: var(--muted); }}
.rid {{ color: var(--muted); overflow-wrap: anywhere; }}
.err {{ margin-left: auto; font-family: var(--mono); font-size: .82rem; }}
blockquote {{ margin: 0; font-size: .9rem; color: var(--fg); overflow-wrap: anywhere; line-height: 1.6; }}
mark.chg {{ background: var(--mark); color: inherit; padding: 0 2px; border-radius: 2px; }}
.br {{ color: var(--muted); }}
table.tags {{ font-size: .85rem; }} table.tags td, table.tags th {{ padding: 5px 8px; }}
.decision {{ display: grid; gap: 10px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px; }}
.decision ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; }}
footer {{ color: var(--muted); font-size: .85rem; }}
.how {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 18px; display: grid; gap: 8px; }}
.how ol {{ margin: 0; padding-left: 22px; display: grid; gap: 8px; max-width: 76ch; }}
.how li::marker {{ font-family: var(--mono); color: var(--accent); font-size: .85em; }}
.how ol ul {{ margin: 6px 0 0; padding-left: 18px; display: grid; gap: 4px; }}
.how .src {{ color: var(--muted); font-size: .82rem; }}
@media (max-width: 560px) {{ h1 {{ font-size: 1.6rem; }} td.bars {{ min-width: 140px; }} }}
</style>
<main>
<div class="intro">
  <span class="eyebrow">Weight-space PII · milestone 7 · for Desmond's review</span>
  <h1>Did Opus 5.5's edit make MiniLM better on Singapore text?</h1>
  <p>Opus 5.5 edited 12 tensors in MiniLM's MLPs during the 50-pass run. This page scores the edited MiniLM on the two hidden sets it never saw, against the unedited baseline and the M5 prediction. Every number comes from <code>runs/p50/eval/evaluation.json</code>, written by <code>scripts/evaluate.py</code>. Tick each item once you have reviewed it. The ticks stay in this browser only.</p>
  <div class="headline">
    <div class="stat"><small>Singapore F1</small><b>{sg['f1']:.3f} → {sg['f1'] + sg['diff']:.3f}</b><span class="delta-good num">{sg['diff']:+.3f} ± {sg['diff_se']:.3f}</span></div>
    <div class="stat"><small>Nemotron F1, all 55 labels</small><b>{nem['f1']:.3f} → {nem['f1'] + nem['diff']:.3f}</b><span class="num">{nem['diff']:+.4f} ± {nem['diff_se']:.4f}, counts as unchanged</span></div>
    <div class="stat"><small>Cue-less NRICs right</small><b>{nric['base_correct']} → {nric['edited_correct']}</b><span class="num">of {nric['entities']}</span></div>
    <div class="stat"><small>Titles wrongly tagged as a name, lower is better</small><b>{titles['groups']['targeted']['base_hit']} → {titles['groups']['targeted']['edited_hit']}</b><span class="num">of {titles['groups']['targeted']['n']:,}</span></div>
  </div>
</div>

<section id="main-table">
  <h2><span class="step">1</span>Main table<label class="check"><input type="checkbox" id="c-main" data-key="main"> Reviewed</label></h2>
  <p>The ± values are bootstrap standard errors over rows. The edited row gives the paired difference against the base, with both models scored on the same rows.</p>
  <div class="tbl"><table>
    <thead><tr><th>Model</th><th>Singapore F1</th><th>Nemotron F1</th><th>NRICs without "tax" right</th><th>Titles wrongly tagged as a name (lower is better)</th><th>Weights changed</th></tr></thead>
    <tbody>
      <tr><th scope="row">Base</th><td class="num">{sg['f1']:.3f} ± {sg['se']:.3f}</td><td class="num">{nem['f1']:.3f} ± {nem['se']:.3f}</td><td class="num">0% (0 of 323)</td><td class="num">{pct(titles['groups']['targeted']['base_share'])} ({titles['groups']['targeted']['base_hit']} of {titles['groups']['targeted']['n']:,})</td><td>none</td></tr>
      <tr><th scope="row">Edited, 50 passes</th><td class="num">{sg['f1'] + sg['diff']:.3f}, {sg['diff']:+.3f} ± {sg['diff_se']:.3f}</td><td class="num">{nem['f1'] + nem['diff']:.3f}, {nem['diff']:+.4f} ± {nem['diff_se']:.4f}</td><td class="num">{pct(nric['edited_share'])} ({nric['edited_correct']} of 323)</td><td class="num">{pct(titles['groups']['targeted']['edited_share'], 1)} ({titles['groups']['targeted']['edited_hit']} of {titles['groups']['targeted']['n']:,})</td><td>12 tensors, 7,304 numbers, 0.03% of MiniLM</td></tr>
    </tbody></table></div>
  <p>Opus 5.5 estimated +0.065 ± 0.005 on its held-back half of the practice set. The clean main-eval gain, {sg['diff']:+.3f}, is close, so tuning on that half inflated its number only a little.</p>
  {HOW_MAIN}
</section>

<section id="categories">
  <h2><span class="step">2</span>Categories against the M5 prediction<label class="check"><input type="checkbox" id="c-kinds" data-key="kinds"> Reviewed</label></h2>
  <p>Share of Singapore main-eval entities tagged with the exact span and the right label. A category is fixed if its share rises by at least 10 points. All eight verdicts match the prediction.</p>
  <div class="legend"><span><i style="background:var(--base)"></i>Unedited MiniLM</span><span><i style="background:var(--accent)"></i>Edited MiniLM</span></div>
  <div class="tbl"><table>
    <thead><tr><th>Category</th><th>Share right</th><th>Change, points</th><th>Predicted</th><th>Result</th></tr></thead>
    <tbody>{kind_rows()}</tbody></table></div>
  <div class="callout">
    <h3>Where the prediction went wrong</h3>
    <p>It expected Singapore F1 to rise by 0.002 to 0.009. It rose by {sg['diff']:.3f}. The title repair mostly removes false positives, which no category counts: false positives fell from 1,196 to 251. And the prediction assumed the 118 NRICs MiniLM never tagged would stay untagged, which capped the fix at 205. The edit fixed {nric['edited_correct']}.</p>
    <p>NRICs with "tax" just before them rose 9.99 points, 622 to 693 of 711, just under the line for fixed.</p>
  </div>
  {HOW_KINDS}
</section>

<section id="mechanism">
  <h2><span class="step">3</span>The two repairs, measured<label class="check"><input type="checkbox" id="c-mech" data-key="mech"> Reviewed</label></h2>
  <p>Both mechanism metrics were defined from Opus 5.5's final answer before any edited score existed. Mr and Mrs are the control, because the final answer says MiniLM already handled them. In the answers, a title is never part of a name, so every name tag on a title is a false positive and a lower share is better. Of the 1,451 targeted titles in the main eval, 1,450 sit outside every gold name. The one inside is a surname, and the metric skips it.</p>
  <div class="tbl"><table>
    <thead><tr><th>Group</th><th>Title</th><th>Occurrences</th><th>Tagged as a name, unedited</th><th>Tagged as a name, edited</th></tr></thead>
    <tbody>{title_rows()}</tbody></table></div>
  <div class="callout good">
    <h3>No sign of new knowledge</h3>
    <p>Name words just after a title went from {pct(by_title['after_title']['base_share'], 1)} to {pct(by_title['after_title']['edited_share'], 1)} right. All other name words went from {pct(by_title['no_title']['base_share'], 1)} to {pct(by_title['no_title']['edited_share'], 1)}. Only 26 of the 212 untagged one-word given names are now found. The gains sit next to the obstacles Opus 5.5 removed, which fits the paper's pattern.</p>
  </div>
  {HOW_MECH}
</section>

<section id="nemotron">
  <h2><span class="step">4</span>The cost on Nemotron<label class="check"><input type="checkbox" id="c-nem" data-key="nem"> Reviewed</label></h2>
  <p>Overall, both Nemotron scores count as unchanged under the prediction's 0.005 rule: {nem['diff']:+.4f} on all labels and {gate['diff']:+.4f} on the two Singapore labels. Each is more than twice its standard error, so it is real but small. The loss sits in the ID labels the prediction named as most at risk.</p>
  <div class="tbl"><table>
    <thead><tr><th>Label</th><th>Base F1</th><th>Edited F1</th><th>Change</th></tr></thead>
    <tbody>{regression_rows()}</tbody></table></div>
  <div class="callout">
    <h3>Why national_id fell</h3>
    <p>MiniLM still gets Nemotron's own national IDs right, 527 against 526. The loss is 57 new false <code>national_id</code> tags. {trans['shapes']['A99999999999']} of them are biometric IDs shaped like a capital letter and 11 digits, such as <code>F98324157683</code>. Opus 5.5 tested look-alike codes of 7 or 8 digits only. That the NRIC detector ignores length is inferred from these shapes, not tested.</p>
  </div>
  {HOW_NEM}
</section>

<section id="changed-rows">
  <h2><span class="step">5</span>Rows the edit changed<label class="check"><input type="checkbox" id="c-rows" data-key="rows"> Reviewed</label></h2>
  <p>Five fixed and five broken rows per set, drawn at random with seed 20261106 from every changed row. In total, {cnt['sg_eval']['fixed']:,} Singapore rows are fixed and {cnt['sg_eval']['broken']} broken. On Nemotron, {cnt['nemotron_eval']['fixed']} are fixed and {cnt['nemotron_eval']['broken']} broken. Highlighted text is where the tags changed. On Singapore text, <code>first_name</code> and <code>last_name</code> both count as <code>name</code>.</p>
  {HOW_ROWS}
  <div class="tabs" role="tablist">
    <button role="tab" id="t-sg-fixed" aria-selected="true" data-panel="p-sg-fixed">Singapore, fixed</button>
    <button role="tab" id="t-sg-broken" aria-selected="false" data-panel="p-sg-broken">Singapore, broken</button>
    <button role="tab" id="t-nm-fixed" aria-selected="false" data-panel="p-nm-fixed">Nemotron, fixed</button>
    <button role="tab" id="t-nm-broken" aria-selected="false" data-panel="p-nm-broken">Nemotron, broken</button>
  </div>
  <div class="panel" id="p-sg-fixed" role="tabpanel">{cards[('sg_eval', 'fixed')]}</div>
  <div class="panel" id="p-sg-broken" role="tabpanel" hidden>{cards[('sg_eval', 'broken')]}</div>
  <div class="panel" id="p-nm-fixed" role="tabpanel" hidden>{cards[('nemotron_eval', 'fixed')]}</div>
  <div class="panel" id="p-nm-broken" role="tabpanel" hidden>{cards[('nemotron_eval', 'broken')]}</div>
</section>

<section id="checks">
  <h2><span class="step">6</span>Checks run before scoring<label class="check"><input type="checkbox" id="c-checks" data-key="checks"> Reviewed</label></h2>
  <p>These close the M7 checks that the M5 and M6 reports left open. Scoring stops if any fails.</p>
  <ul class="checks">{check_rows()}</ul>
  {HOW_CHECKS}
</section>

<section id="decide">
  <h2><span class="step">7</span>Your call</h2>
  <div class="decision">
    <p>Once items 1 to 6 are reviewed, the options are:</p>
    <ul>
      <li><b>Approve M7 and start M8.</b> M8 writes the write-up and builds the demo, a fictional Singapore claims letter highlighted by both models side by side.</li>
      <li><b>Test the length question first.</b> Run the edited MiniLM on one sentence with codes of a letter and 7 to 12 digits, to confirm why Nemotron's <code>national_id</code> fell. This doesn't change the frozen edit.</li>
      <li><b>Ask for changes to the M7 report</b> before approving.</li>
    </ul>
    <p class="n">Reply in the Claude session with your choice.</p>
  </div>
</section>

<footer>Built by <code>scripts/m7_review_page.py</code> from <code>runs/p50/eval/</code>. The report is <code>docs/m7_report.md</code>, and the full changed-row list is <code>runs/p50/eval/changed_rows.jsonl</code>.</footer>
</main>
<script>
(() => {{
  const tabs = [...document.querySelectorAll('.tabs button')];
  tabs.forEach(b => b.addEventListener('click', () => {{
    tabs.forEach(t => {{ const on = t === b; t.setAttribute('aria-selected', on); document.getElementById(t.dataset.panel).hidden = !on; }});
  }}));
  const store = {{
    get(k) {{ try {{ return localStorage.getItem('m7-review-' + k) === '1'; }} catch (e) {{ return false; }} }},
    set(k, v) {{ try {{ localStorage.setItem('m7-review-' + k, v ? '1' : '0'); }} catch (e) {{}} }}
  }};
  document.querySelectorAll('.check input').forEach(c => {{
    c.checked = store.get(c.dataset.key);
    c.addEventListener('change', () => store.set(c.dataset.key, c.checked));
  }});
}})();
</script>
"""
OUT.write_text(html)
print(f"wrote {OUT} ({len(html):,} bytes)")

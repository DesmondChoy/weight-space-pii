"""M4 checks: save_edit refusals through the sandboxed kernel, the server's own re-check, the server against
an independent predict + grade + summarize run, and the forward-pass array files. Writes runs/m4_checks.json."""

import hashlib
import json
import random
import sys

import numpy as np

from wsp.data import MAX_LEN, MODEL_DIR, read_jsonl
from wsp.harness.box import MODEL_FILES, RUNS_ROOT, setup_box
from wsp.harness.kernel_host import Kernel
from wsp.harness.server import RequestError, Server
from wsp.predict import load, predict
from wsp.score import FAILURES, SG_LABELS, grade, summarize

SEED = 20261105
N_ITEMS = 50
MLP = "bert.encoder.layer.0.intermediate.dense.weight"
BYPASS = "bert.encoder.layer.1.output.dense.weight"
OWN_TEXT = "Hi, I'm Tan Wei Ming s/o Rajan, NRIC S1234567D, call me at 9123 4567."

REFUSAL_CASES = (
    ("embeddings", "bert.embeddings.word_embeddings.weight", "w = load(NAME); w[100, 0] += 0.5", "embeddings"),
    ("classification head", "classifier.weight", "w = load(NAME); w[3, 0] += 0.5", "classification head"),
    ("whole-tensor constant multiple", MLP, "w = 1.1 * load(NAME)", "c=1.1"),
    ("near multiple: 2x, one entry nudged", MLP, "w = 2 * load(NAME); w[0, 0] += 1e-3", "almost only multiplies"),
    ("unchanged tensor", MLP, "w = load(NAME)", "unchanged"),
    ("shape mismatch", MLP, "w = load(NAME).T", "shape"),
    ("non-finite value", MLP, "w = load(NAME); w[0, 0] = float('inf')", "not finite"),
    ("unknown name", "bert.encoder.layer.6.output.dense.weight", "w = load(MLP_NAME)", "not a tensor"),
    ("accepted: one entry changed", MLP, "w = load(NAME); w[0, 0] += 0.5", None),
)


def file_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refusal_checks(kernel: Kernel) -> list[dict]:
    rows = []
    for case, name, make, expect in REFUSAL_CASES:
        code = (f"NAME, MLP_NAME = {name!r}, {MLP!r}\n{make}\ntry:\n    save_edit(NAME, w)\n    print('ACCEPTED', edited())\n"
                "except ValueError as e:\n    print(e)")
        out = kernel.run(code).output.strip()
        if expect is None:
            ok = out.startswith("ACCEPTED") and MLP in out
        else:
            ok = out.startswith("save_edit refused") and expect in out
        rows.append({"case": case, "tensor": name, "output": out, "ok": ok})
    return rows


def bypass_check(kernel: Kernel, server: Server, box) -> dict:
    out = kernel.run(f"import numpy as np\nnp.save({str(box.edited / (BYPASS + '.npy'))!r}, (2 * load({BYPASS!r})).numpy())\n"
                     "edited()").output.strip()
    try:
        server.handle(0, server.parse(json.dumps({"model": "edited", "texts": ["a"]})))
        reason = None
    except RequestError as e:
        reason = str(e)
    kernel.run(f"drop_edit({BYPASS!r}); drop_edit({MLP!r})")
    return {"kernel_edited": out, "server_refusal": reason, "edited_after_cleanup": kernel.run("edited()").output.strip(),
            "ok": reason is not None and BYPASS in reason and "c=2" in reason}


def pred_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def scorer_check(server: Server, box, docs, tok, model) -> dict:
    texts = [d.text for d in docs]
    spans = predict(tok, model, texts)
    grades = [grade(d, s, SG_LABELS) for d, s in zip(docs, spans)]
    want_summary = summarize(grades)
    want_pred = [[{"start": s.start, "end": s.end, "label": s.label, "text": d.text[s.start:s.end]} for s in ss]
                 for d, ss in zip(docs, spans)]
    want_grade = [{"by_label": {lab: {"tp": t.tp, "fp": t.fp, "fn": t.fn} for lab, t in g.by_label.items()},
                   "failures": {k: g.failures[k] for k in FAILURES}}
                  for g in grades]
    out = {}
    for k, mode in ((1, "original"), (2, "both")):
        summary = server.handle(k, server.parse(json.dumps({"model": mode, "items": [d.id for d in docs]}))).summary
        rows = pred_rows(box.work / f"pred_{k}.jsonl")
        models = ("original", "edited") if mode == "both" else ("original",)
        res = {"summary_equal": all(summary[m] == want_summary for m in models),
               "spans_equal": all([r["pred"] for r in rows if r["model"] == m] == want_pred for m in models),
               "grades_equal": all([r["grade"] for r in rows if r["model"] == m] == want_grade for m in models),
               "ids_in_order": all([r["id"] for r in rows if r["model"] == m] == [d.id for d in docs] for m in models),
               "overall": summary["original"]["overall"], "failures": summary["original"]["failures"]}
        if mode == "both":
            res["diff"] = summary["diff"]
            res["diff_zero"] = summary["diff"] == {"f1_edited_minus_original": 0.0, "se": 0.0,
                                                   "items_better": 0, "items_worse": 0}
            res["edited_tensors"] = summary["edited_tensors"]
        res["ok"] = all(v for key, v in res.items() if key in ("summary_equal", "spans_equal", "grades_equal",
                                                                "ids_in_order", "diff_zero"))
        out[mode] = res
    out["n_items"] = len(docs)
    out["n_entities_predicted"] = sum(len(s) for s in spans)
    out["ok"] = out["original"]["ok"] and out["both"]["ok"]
    return out


def arrays_check(server: Server, box, docs, tok, model) -> dict:
    texts = [d.text for d in docs] + [OWN_TEXT]
    req = {"model": "original", "items": [d.id for d in docs], "texts": [OWN_TEXT],
           "return": {"label_probs": True, "hidden_states": list(range(7)), "mlp_act": list(range(6)),
                      "attention": list(range(6))}}
    summary = server.handle(3, server.parse(json.dumps(req))).summary
    d = box.work / "fwd_3"
    meta = json.loads((d / "meta.json").read_text())
    lengths = [len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts]
    n = sum(lengths)
    want = {"label_probs.npy": [n, 111], **{f"hidden_L{l}.npy": [n, 384] for l in range(7)},
            **{f"mlp_L{l}.npy": [n, 1536] for l in range(6)},
            **{f"attention_L{l}/{i}.npy": [12, lengths[i], lengths[i]] for l in range(6) for i in range(len(texts))}}
    on_disk = {name: list(np.load(d / name, mmap_mode="r").shape) for name in want if (d / name).exists()}
    probs = np.load(d / "label_probs.npy")
    rows_match = True
    for t in meta["texts"]:
        enc = tok(texts[t["index"]], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ref = model(**enc).logits[0].softmax(-1).numpy()
        rows_match &= bool(np.allclose(probs[t["start_row"]:t["start_row"] + t["length"]], ref, atol=1e-5))
    meta_texts = [{k: t[k] for k in ("index", "id", "start_row", "length")} for t in meta["texts"]]
    want_meta = [{"index": i, "id": ([x.id for x in docs] + [None])[i], "start_row": sum(lengths[:i]),
                  "length": lengths[i]} for i in range(len(texts))]
    both = server.handle(4, server.parse(json.dumps({"model": "both", "texts": [OWN_TEXT],
                                                     "return": {"label_probs": True, "attention": [0]}}))).summary
    res = {"files": {k: v for k, v in on_disk.items() if "/" not in k},
           "attention_files": sum("/" in k for k in on_disk), "attention_files_expected": 6 * len(texts),
           "shapes_ok": on_disk == want, "meta_ok": meta_texts == want_meta and meta["arrays"] == want,
           "meta_has_tokens": all(len(t["tokens"]) == t["length"] == len(t["word_ids"]) for t in meta["texts"]),
           "label_probs_rows_match_unpadded_run": rows_match,
           "probs_sum_to_1": bool(np.allclose(probs.sum(-1), 1, atol=1e-5)),
           "summary_arrays": summary["arrays"], "both_files": sorted(p.name for p in (box.work / "fwd_4").iterdir())}
    res["both_ok"] = res["both_files"] == ["edited_attention_L0", "edited_label_probs.npy", "meta.json",
                                           "original_attention_L0", "original_label_probs.npy"] and "arrays" in both
    res["ok"] = all(res[k] for k in ("shapes_ok", "meta_ok", "meta_has_tokens", "label_probs_rows_match_unpadded_run",
                                     "probs_sum_to_1", "both_ok"))
    return res


def main() -> int:
    RUNS_ROOT.mkdir(exist_ok=True)
    box = setup_box("m4-checks", fresh=True)
    copies = {f: file_sha(box.model / f) == file_sha(MODEL_DIR / f) for f in MODEL_FILES}
    practice = read_jsonl(box.practice_file)
    docs = random.Random(SEED).sample(practice, N_ITEMS)
    kernel = Kernel(box, RUNS_ROOT / "m4_checks.kernel.log")
    kernel.start()
    try:
        server = Server(box, practice)
        refusals = refusal_checks(kernel)
        bypass = bypass_check(kernel, server, box)
    finally:
        kernel.close()
    tok, model = load(MODEL_DIR)
    scorer = scorer_check(server, box, docs, tok, model)
    arrays = arrays_check(server, box, docs[:3], tok, model)
    result = {"seed": SEED, "box": str(box.root), "model_copies_identical": copies, "refusals": refusals,
              "bypass": bypass, "server_vs_scorer": scorer, "arrays": arrays}
    result["all_ok"] = (all(copies.values()) and all(r["ok"] for r in refusals) and bypass["ok"] and scorer["ok"]
                        and arrays["ok"])
    (RUNS_ROOT / "m4_checks.json").write_text(json.dumps(result, indent=1) + "\n")

    print(f"box {box.root}; model copies identical to data/model: {all(copies.values())}")
    print("\nsave_edit through the sandboxed kernel:")
    for r in refusals:
        print(f"  {'ok ' if r['ok'] else 'BAD'} {r['case']:<32} {r['output'][:110]}")
    print(f"\nserver re-check of a file written around save_edit: {'ok' if bypass['ok'] else 'BAD'}")
    print(f"  {bypass['server_refusal']}")
    print(f"  edited() after cleanup: {bypass['edited_after_cleanup']}")
    print(f"\nserver vs independent predict + grade + summarize on {N_ITEMS} practice items (seed {SEED}):")
    for mode in ("original", "both"):
        s = scorer[mode]
        flags = ", ".join(f"{k}={s[k]}" for k in ("summary_equal", "spans_equal", "grades_equal", "ids_in_order"))
        print(f"  {'ok ' if s['ok'] else 'BAD'} {mode:<8} {flags}")
        if mode == "both":
            print(f"      diff {s['diff']}, edited_tensors {s['edited_tensors']}")
    print(f"  overall {scorer['original']['overall']}")
    print(f"  failures {scorer['original']['failures']}")
    print(f"\nforward arrays, 3 items + 1 own text, every kind: {'ok' if arrays['ok'] else 'BAD'}")
    for k in ("shapes_ok", "meta_ok", "meta_has_tokens", "label_probs_rows_match_unpadded_run", "probs_sum_to_1",
              "both_ok"):
        print(f"  {k}={arrays[k]}")
    print(f"  attention files {arrays['attention_files']} of {arrays['attention_files_expected']}")
    for name, shape in arrays["files"].items():
        print(f"  {name:<18} {shape}")
    print(f"  'both' files: {arrays['both_files']}")
    print(f"\nall ok: {result['all_ok']}; wrote {RUNS_ROOT / 'm4_checks.json'}")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

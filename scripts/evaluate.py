"""M7: check the p50 run, score the edited MiniLM on the locked sets, and compare it with the baseline and the prediction.

Needs runs/base/ from scripts/baseline.py and runs/p50/ from the M6 run. Writes runs/p50/eval/: pred_<set>.jsonl,
changed_rows.jsonl, and evaluation.json. Stops with an error if any check fails.
"""

import importlib.util
import json
import platform
import re
import subprocess
import time
from collections import Counter

import torch
import transformers
from safetensors.numpy import load_file

from wsp.data import ROOT, SETS, Span, read_jsonl
from wsp.harness.edit import apply_npz, sha256
from wsp.lock import LOCK, check, sha256_file
from wsp.outcomes import outcomes
from wsp.predict import load, predict
from wsp.score import NAME_LABELS, bootstrap, grade, split_names, summarize

_spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts" / "baseline.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)
SCORINGS, kind_table = baseline.SCORINGS, baseline.kind_table

BASE = ROOT / "runs" / "base"
RUN = ROOT / "runs" / "p50"
FORWARD = RUN / "forward"
OUT = RUN / "eval"
ORIGINAL = ROOT / "data" / "model" / "model.safetensors"
FROZEN_HARNESS = {"lock.json", "pyproject.toml", "uv.lock", "scripts/run_editor.py"}
TITLE_GROUPS = {"targeted": ("Madame", "Master", "Miss", "Mstr", "Mayoress"), "control": ("Mr", "Mrs"),
                "untargeted": ("Mdm", "Dr", "Mx", "Madam")}
TITLE = re.compile(r"\b(" + "|".join(w for g in TITLE_GROUPS.values() for w in g) + r")\b")
AFTER_TITLE = re.compile(r"\.?\s+")
TITLE_BEFORE = re.compile(TITLE.pattern + r"\.?\s+$")
REGRESSION = SCORINGS["nemotron_regression"][1]


def shape(value: str) -> str:
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", value)))
FIXED_PP = 10
NEMOTRON_NEGLIGIBLE = 0.005

checks: list[dict] = []


def record(name: str, ok: bool, detail=None) -> None:
    checks.append({"name": name, "passed": bool(ok), "detail": detail})
    print(f"{'pass' if ok else 'FAIL'}  {name}" + (f"  {detail}" if not ok and detail is not None else ""))


def require_all() -> None:
    if failed := [c["name"] for c in checks if not c["passed"]]:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "evaluation.json").write_text(json.dumps({"checks": checks}, indent=1) + "\n")
        raise SystemExit(f"checks failed: {failed}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def strip_session_id(cmd: list[str]) -> list[str]:
    i = cmd.index("--session-id")
    return cmd[:i] + cmd[i + 2:]


def tensor_list(manifest: dict) -> list[tuple]:
    return sorted((t["name"], t["sha256_original"], t["sha256_edited"]) for t in manifest["tensors"])


def check_tensors(label: str, npz_path, manifest: dict, originals: dict) -> dict:
    """Rebuild one npz against the original weights and compare names and hashes with its manifest."""
    tensors = apply_npz(npz_path, ORIGINAL)
    want = {t["name"]: t for t in manifest["tensors"]}
    names_ok = set(tensors) == set(want)
    edited_ok = names_ok and all(sha256(tensors[n]) == want[n]["sha256_edited"] for n in want)
    original_ok = names_ok and all(sha256(originals[n]) == want[n]["sha256_original"] for n in want)
    record(f"{label}: names, original and edited hashes match its manifest", names_ok and edited_ok and original_ok,
           {"names": names_ok, "edited": edited_ok, "original": original_ok})
    return tensors


def run_checks(lock: dict) -> dict:
    lock_sha = sha256_file(LOCK)
    baseline_json = json.loads((BASE / "baseline.json").read_text())
    run = json.loads((RUN / "run.json").read_text())
    record("lock.json matches the tree", True)
    record("lock sha256 equals baseline.json and run.json", lock_sha == baseline_json["lock_sha256"] == run["lock_sha256"],
           [lock_sha, baseline_json["lock_sha256"], run["lock_sha256"]])

    record("run.json: git tree was clean", run["git"]["dirty"] is False, run["git"])
    record("run.json: ended by final_answer", run["ended_by"] == "final_answer", run["ended_by"])
    record("run.json: no edit_error", run["edit_error"] is None, run["edit_error"])
    settings = lock["run"]
    same = {"cli_command": strip_session_id(run["cli_command"]) == settings["cli_command"],
            **{k: run[k] == settings[k] for k in ("cli_env", "cli_env_passed", "cli_version", "budget", "max_turns")},
            "name": RUN.name == settings["name"]}
    record("run.json settings equal lock run settings", all(same.values()), same)

    freeze = git("log", "-1", "--format=%H", "--", "lock.json")
    changed = git("diff", "--name-only", freeze, run["git"]["head"]).splitlines()
    touched = [p for p in changed if p.startswith("src/") or p in FROZEN_HARNESS]
    record("harness code unchanged between the freeze and the run's head", not touched,
           {"freeze": freeze, "head": run["git"]["head"], "changed_paths": changed, "harness_paths": touched})

    base_ok = {}
    for set_name, want in baseline_json["pred_sha256"].items():
        rows = [json.loads(line) for line in (BASE / f"pred_{set_name}.jsonl").open()]
        ids = [r["id"] for r in rows]
        base_ok[set_name] = (sha256_file(BASE / f"pred_{set_name}.jsonl") == want
                             and ids == [d.id for d in read_jsonl(SETS / f"{set_name}.jsonl")])
    record("base predictions: sha256 match and same ids in the same order as the sets", all(base_ok.values()), base_ok)

    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "device": "cpu"}
    want_env = baseline_json["environment"]
    record("environment equals baseline.json", all(env[k] == want_env[k] for k in env),
           {k: [env[k], want_env[k]] for k in env if env[k] != want_env[k]})
    env["torch_threads"] = torch.get_num_threads()
    checks[-1]["detail"] = {**env, "baseline_torch_threads": want_env["torch_threads"]}

    originals = {k: v.astype("float32") for k, v in load_file(ORIGINAL).items()}
    submitted = json.loads((RUN / "edit" / "manifest.json").read_text())
    tensors = check_tensors("submitted edit", RUN / "edit" / "edit.npz", submitted, originals)

    submitted_list = tensor_list(submitted)
    passes, cache = {}, {}
    for k in range(1, run["budget"] + 1):
        manifest = json.loads((FORWARD / str(k) / "edit_manifest.json").read_text())
        npz = manifest["npz"]
        if npz is not None and npz not in cache:
            cache[npz] = check_tensors(f"forward pass {k} ({npz})", FORWARD / npz, manifest, originals)
        passes[k] = {"npz": npz, "n_tensors": len(manifest["tensors"]),
                     "equals_submitted": tensor_list(manifest) == submitted_list}
    record("forward passes 1..50 each have a manifest", sorted(passes) == list(range(1, run["budget"] + 1)))
    return {"baseline": baseline_json, "run": run, "tensors": tensors, "passes": passes, "environment": env,
            "freeze": freeze, "harness_diff": changed}


def edited_model(tensors: dict):
    tok, model = load()
    state = model.state_dict()
    for name, arr in tensors.items():
        assert name in state and tuple(state[name].shape) == arr.shape, f"{name} does not fit the model"
        state[name].copy_(torch.from_numpy(arr))
    return tok, model


def spans_of(rows: list) -> list[list[Span]]:
    return [[Span(s["start"], s["end"], s["label"]) for s in r["spans"]] for r in rows]


def key(spans) -> list[tuple]:
    return sorted((s.start, s.end, s.label) for s in spans)


def server_agreement(docs, base_preds, edited_preds) -> dict:
    by_id = {d.id: i for i, d in enumerate(docs)}
    mismatches = Counter()
    rows = [json.loads(line) for line in (FORWARD / "50" / "pred.jsonl").open()]
    for r in rows:
        i = by_id[r["id"]]
        mine = base_preds[i] if r["model"] == "original" else edited_preds[i]
        server = sorted((p["start"], p["end"], p["label"]) for p in r["pred"])
        mismatches[r["model"], key(mine) == server] += 1
    result = {m: {"matches": mismatches[m, True], "mismatches": mismatches[m, False]} for m in ("original", "edited")}
    record("forward pass 50 server predictions equal ours (original vs base, edited vs edited)",
           all(v["mismatches"] == 0 and v["matches"] > 0 for v in result.values()), result)
    return result


def verdict(pp: float) -> str:
    return "fixed" if pp >= FIXED_PP else "broken" if pp <= -FIXED_PP else "unchanged"


def nemotron_verdict(b: dict) -> str:
    if abs(b["diff"]) <= NEMOTRON_NEGLIGIBLE or abs(b["diff"]) <= 2 * b["diff_se"]:
        return "unchanged"
    return "better" if b["diff"] > 0 else "worse"


def kind_verdicts(base: dict, edited: dict) -> dict:
    out = {}
    for k, b in base.items():
        e = edited[k]
        b_share, e_share = b["correct"] / b["entities"], e["correct"] / e["entities"]
        pp = (e_share - b_share) * 100
        out[k] = {"entities": b["entities"], "base_correct": b["correct"], "edited_correct": e["correct"],
                  "base_share": round(b_share, 4), "edited_share": round(e_share, 4), "change_pp": round(pp, 2),
                  "verdict": verdict(pp)}
    return out


def title_metrics(docs, base_preds, edited_preds) -> dict:
    per_word = {w: {"n": 0, "base_hit": 0, "edited_hit": 0} for g in TITLE_GROUPS.values() for w in g}
    follow = {"n": 0, "base_right": 0, "edited_right": 0}
    for d, bp, ep in zip(docs, base_preds, edited_preds, strict=True):
        gold_words = {w.start: w for w in split_names(d.text, [s for s in d.spans if s.label in NAME_LABELS])}
        right = [{(o.word.start, o.word.end) for o in outcomes(d, p) if o.correct} for p in (bp, ep)]
        for m in TITLE.finditer(d.text):
            if any(g.start < m.end() and m.start() < g.end for g in d.spans):
                continue
            row = per_word[m.group(1)]
            row["n"] += 1
            for name, preds in (("base_hit", bp), ("edited_hit", ep)):
                row[name] += any(p.label in NAME_LABELS and p.start < m.end() and m.start() < p.end for p in preds)
            if m.group(1) in TITLE_GROUPS["targeted"] and (gap := AFTER_TITLE.match(d.text, m.end())):
                if word := gold_words.get(gap.end()):
                    follow["n"] += 1
                    follow["base_right"] += (word.start, word.end) in right[0]
                    follow["edited_right"] += (word.start, word.end) in right[1]

    def shares(row: dict) -> dict:
        n = row["n"]
        return {**row, "base_share": round(row["base_hit"] / n, 4) if n else None,
                "edited_share": round(row["edited_hit"] / n, 4) if n else None}

    groups = {g: shares({f: sum(per_word[w][f] for w in words) for f in ("n", "base_hit", "edited_hit")})
              for g, words in TITLE_GROUPS.items()}
    n = follow["n"]
    return {"groups": groups, "words": {g: {w: shares(per_word[w]) for w in words} for g, words in TITLE_GROUPS.items()},
            "name_after_targeted_title": {**follow, "base_share": round(follow["base_right"] / n, 4) if n else None,
                                          "edited_share": round(follow["edited_right"] / n, 4) if n else None}}


def names_by_title(docs, base_preds, edited_preds) -> dict:
    """Name words split by whether any title word sits just before their name. A gain only after titles is the
    edit removing the title's confusion; a gain elsewhere would be names MiniLM newly recognizes."""
    out = {c: Counter() for c in ("after_title", "no_title")}
    for d, bp, ep in zip(docs, base_preds, edited_preds, strict=True):
        edited = {(o.word.start, o.word.end): o for o in outcomes(d, ep)}
        for o in outcomes(d, bp):
            if o.want != "name":
                continue
            c = out["after_title" if TITLE_BEFORE.search(d.text[max(0, o.gold.start - 20):o.gold.start]) else "no_title"]
            e = edited[(o.word.start, o.word.end)]
            c["n"] += 1
            c["base_right"] += o.correct
            c["edited_right"] += e.correct
            if o.correct != e.correct:
                c[f"fixed from {o.got}" if e.correct else f"broken to {e.got}"] += 1
    return {ctx: {"n": c["n"], "base_share": round(c["base_right"] / c["n"], 4),
                  "edited_share": round(c["edited_right"] / c["n"], 4),
                  "changes": dict(Counter({k: v for k, v in c.items() if k.startswith(("fixed", "broken"))}).most_common())}
            for ctx, c in out.items()}


def regression_transitions(docs, base_preds, edited_preds, top: int = 15) -> list[dict]:
    """Gold Nemotron entities whose outcome changed, grouped by gold label and the base and edited outcome, with
    the shapes of their values (digits as 9, capitals as A, lower case as a)."""
    groups: dict[tuple, Counter] = {}
    for d, bp, ep in zip(docs, base_preds, edited_preds, strict=True):
        edited = {(o.word.start, o.word.end): o.got for o in outcomes(d, ep, REGRESSION)}
        for o in outcomes(d, bp, REGRESSION):
            if (e := edited[(o.word.start, o.word.end)]) != o.got:
                groups.setdefault((o.want, o.got, e), Counter())[shape(d.value(o.word))] += 1
    rows = [{"gold": g, "base": b, "edited": e, "n": sum(s.values()), "shapes": dict(s.most_common(4))}
            for (g, b, e), s in groups.items()]
    return sorted(rows, key=lambda r: -r["n"])[:top]


def changed_rows(set_name, docs, base_preds, edited_preds, base_grades, edited_grades) -> tuple[list, Counter]:
    rows, counts = [], Counter()
    for i, (d, bp, ep, bg, eg) in enumerate(zip(docs, base_preds, edited_preds, base_grades, edited_grades, strict=True)):
        b_err, e_err = bg.total.fp + bg.total.fn, eg.total.fp + eg.total.fn
        b_key, e_key = key(bp), key(ep)
        if e_err < b_err:
            status = "fixed"
        elif e_err > b_err:
            status = "broken"
        elif b_key != e_key:
            status = "mixed"
        else:
            continue
        counts[status] += 1
        rows.append({"set": set_name, "position": i, "id": d.id, "status": status,
                     "base": {"tp": bg.total.tp, "fp": bg.total.fp, "fn": bg.total.fn},
                     "edited": {"tp": eg.total.tp, "fp": eg.total.fp, "fn": eg.total.fn},
                     "base_only_spans": [list(s) for s in b_key if s not in e_key],
                     "edited_only_spans": [list(s) for s in e_key if s not in b_key]})
    return rows, counts


def regression_by_label(base_summary: dict, edited_summary: dict) -> list[dict]:
    rows = [{"label": lab, "base_f1": b["f1"], "edited_f1": edited_summary["by_label"][lab]["f1"],
             "change": round(edited_summary["by_label"][lab]["f1"] - b["f1"], 4)}
            for lab, b in base_summary["by_label"].items()]
    return sorted(rows, key=lambda r: r["change"])


def main() -> None:
    start = time.time()
    lock = check()
    ctx = run_checks(lock)
    require_all()
    OUT.mkdir(parents=True, exist_ok=True)

    tok, model = edited_model(ctx["tensors"])
    docs, base_preds, edited_preds = {}, {}, {}
    for set_name in dict.fromkeys(s for s, _ in SCORINGS.values()):
        docs[set_name] = read_jsonl(SETS / f"{set_name}.jsonl")
        base_preds[set_name] = spans_of([json.loads(line) for line in (BASE / f"pred_{set_name}.jsonl").open()])
        edited_preds[set_name] = predict(tok, model, [d.text for d in docs[set_name]])
        with (OUT / f"pred_{set_name}.jsonl").open("w") as f:
            for d, pred in zip(docs[set_name], edited_preds[set_name], strict=True):
                f.write(json.dumps({"id": d.id, "spans": [vars(s) for s in pred]}, ensure_ascii=False) + "\n")
        print(f"predicted {set_name}: {len(docs[set_name])} rows")

    agreement = server_agreement(docs["sg_train"], base_preds["sg_train"], edited_preds["sg_train"])
    require_all()

    scores, grades = {}, {}
    for name, (set_name, labels) in SCORINGS.items():
        bg = [grade(d, p, labels) for d, p in zip(docs[set_name], base_preds[set_name], strict=True)]
        eg = [grade(d, p, labels) for d, p in zip(docs[set_name], edited_preds[set_name], strict=True)]
        grades[name] = (bg, eg)
        boot = bootstrap(bg, eg)
        scores[name] = {"set": set_name, "rows": len(bg), "labels": len(labels), "base": summarize(bg),
                        "edited": summarize(eg), "bootstrap": boot}
        if name.startswith("nemotron"):
            scores[name]["verdict"] = nemotron_verdict(boot)
        print(f"{name}: base F1 {boot['f1']:.4f}, edited {boot['f1'] + boot['diff']:.4f}, diff {boot['diff']:+.4f} ± {boot['diff_se']:.4f}"
              + (f", {scores[name]['verdict']}" if "verdict" in scores[name] else ""))

    base_kinds = kind_table(docs["sg_eval"], base_preds["sg_eval"])
    edited_kinds = kind_table(docs["sg_eval"], edited_preds["sg_eval"])
    kinds = kind_verdicts(base_kinds, edited_kinds)
    for k, v in kinds.items():
        print(f"{k:38} {v['base_share']:.4f} -> {v['edited_share']:.4f} ({v['change_pp']:+.2f} pp) {v['verdict']}")
    nric = {k: kinds[k] for k in ("national_id/no tax context", "national_id/tax context")}
    titles = title_metrics(docs["sg_eval"], base_preds["sg_eval"], edited_preds["sg_eval"])
    for g, v in titles["groups"].items():
        print(f"titles {g}: n {v['n']}, name-tagged {v['base_share']} -> {v['edited_share']}")
    print(f"name after targeted title: {titles['name_after_targeted_title']}")
    by_title = names_by_title(docs["sg_eval"], base_preds["sg_eval"], edited_preds["sg_eval"])
    for where, v in by_title.items():
        print(f"names {where}: n {v['n']}, right {v['base_share']} -> {v['edited_share']}")
    transitions = regression_transitions(docs["nemotron_eval"], base_preds["nemotron_eval"],
                                         edited_preds["nemotron_eval"])
    for t in transitions[:6]:
        print(f"nemotron {t['gold']}: {t['base']} -> {t['edited']} x{t['n']} {t['shapes']}")

    changed, change_counts = [], {}
    for scoring, set_name in (("singapore", "sg_eval"), ("nemotron_regression", "nemotron_eval")):
        rows, counts = changed_rows(set_name, docs[set_name], base_preds[set_name], edited_preds[set_name], *grades[scoring])
        changed += rows
        change_counts[set_name] = {s: counts[s] for s in ("fixed", "broken", "mixed")}
    with (OUT / "changed_rows.jsonl").open("w") as f:
        for r in changed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"changed rows: {change_counts}")

    result = {"lock_sha256": sha256_file(LOCK), "checks": checks, "environment": ctx["environment"],
              "freeze_commit": ctx["freeze"], "harness_diff_paths": ctx["harness_diff"],
              "forward_passes": ctx["passes"], "server_agreement": agreement,
              "pred_sha256": {s: sha256_file(OUT / f"pred_{s}.jsonl") for s in edited_preds},
              "scores": scores,
              "nemotron_regression_by_label": regression_by_label(scores["nemotron_regression"]["base"],
                                                                  scores["nemotron_regression"]["edited"]),
              "kinds": {"base": base_kinds, "edited": edited_kinds, "verdicts": kinds},
              "mechanism": {"nric": nric, "titles": titles}, "names_by_title": by_title,
              "nemotron_transitions": transitions,
              "changed_row_counts": change_counts, "seconds": round(time.time() - start, 1)}
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {(OUT / 'evaluation.json').relative_to(ROOT)} in {result['seconds']}s")


if __name__ == "__main__":
    main()

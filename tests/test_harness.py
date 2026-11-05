import json
import os
import stat
import subprocess
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.numpy import load_file

from wsp.data import MODEL_DIR, ROOT, SETS, read_jsonl
from wsp.harness.box import SANDBOX_ROOT, Box, _rmtree, setup_box
from wsp.harness.driver import Editor, EditorReply, Final, Forward, Invalid, Python, parse_reply, run
from wsp.harness.edit import apply_npz, read_edits, sha256, write_edit
from wsp.harness.kernel import refusal
from wsp.harness.kernel_host import Kernel
from wsp.harness.server import Request, RequestError, Server, hf_name, parse_request
from wsp.lock import sha256_file

WEIGHTS = MODEL_DIR / "model.safetensors"
PRACTICE = read_jsonl(SETS / "sg_train.jsonl")
IDS = {d.id for d in PRACTICE}
MLP = "bert.encoder.layer.0.intermediate.dense.weight"
GAMMA = "bert.encoder.layer.2.output.LayerNorm.gamma"
ATTN_OUT = "bert.encoder.layer.4.attention.output.dense.weight"
BIAS = "bert.encoder.layer.2.output.dense.bias"
TEST_BOXES = ("unittest-kernel", "unittest-server", "unittest-driver", "unittest-limit", "unittest-proto",
              "unittest-setup")


def tearDownModule():
    for name in TEST_BOXES:
        _rmtree(SANDBOX_ROOT / name)


class Refusals(unittest.TestCase):
    orig = torch.linspace(-1, 1, 12).reshape(3, 4)

    def check(self, name, new, orig=orig):
        return refusal(name, new, orig)

    def test_embeddings(self):
        self.assertIn("embeddings", self.check("bert.embeddings.word_embeddings.weight", self.orig + 1))

    def test_classifier_head(self):
        self.assertIn("classification head", self.check("classifier.weight", self.orig + 1))

    def test_constant_multiple(self):
        self.assertIn("c=2", self.check(MLP, 2 * self.orig))

    def test_unchanged_tensor_is_a_constant_multiple_of_one(self):
        self.assertIn("unchanged", self.check(MLP, self.orig.clone()))

    def test_shape(self):
        self.assertIn("shape", self.check(MLP, self.orig.reshape(4, 3)))

    def test_non_finite(self):
        new = self.orig.clone()
        new[0, 0] = float("nan")
        self.assertIn("not finite", self.check(MLP, new))

    def test_unknown_name(self):
        self.assertIn("not a tensor", self.check("bert.encoder.layer.9.foo", self.orig, None))

    def test_one_changed_entry_is_accepted(self):
        new = self.orig.clone()
        new[1, 2] += 0.5
        self.assertIsNone(self.check(MLP, new))


class NearMultiples(unittest.TestCase):
    """The constant-multiple rule on real MiniLM tensors: disguised rescales are refused, targeted edits pass."""

    @classmethod
    def setUpClass(cls):
        weights = load_file(str(MODEL_DIR / "model.safetensors"))
        cls.w = torch.from_numpy(weights[ATTN_OUT])
        cls.b = torch.from_numpy(weights[BIAS])

    def refused(self, name, orig, new):
        return refusal(name, new, orig)

    def test_disguised_rescales_are_refused(self):
        torch.manual_seed(0)
        nudged = 2 * self.w
        nudged[0, 0] += 1e-3
        all_but_one_row = 2 * self.w
        all_but_one_row[0] = self.w[0]
        cases = {"2x plus one nudged entry": nudged,
                 "2x plus tiny noise": 2 * self.w + 1e-3 * self.w.std() * torch.randn_like(self.w),
                 "2x every row but one": all_but_one_row}
        for label, new in cases.items():
            with self.subTest(label):
                self.assertIn("almost only multiplies", self.refused(ATTN_OUT, self.w, new) or "")
        bias = 2 * self.b
        bias[0] += 1e-3
        self.assertIn("almost only multiplies", self.refused(BIAS, self.b, bias) or "")

    def test_targeted_edits_pass(self):
        torch.manual_seed(0)
        head_off = self.w.clone()
        head_off[:, :64] = 0
        cols = self.w.clone()
        cols[:, :64] *= 2
        u = torch.randn(self.w.shape[0])
        u /= u.norm()
        entry = self.w.clone()
        entry[5, 7] += 0.5
        half_bias = self.b.clone()
        half_bias[:64] *= 2
        cases = {"zero one head's 64 columns": (ATTN_OUT, self.w, head_off),
                 "2x 64 columns": (ATTN_OUT, self.w, cols),
                 "remove one direction": (ATTN_OUT, self.w, self.w - 1.5 * torch.outer(u, u @ self.w)),
                 "change one entry": (ATTN_OUT, self.w, entry),
                 "2x 64 bias entries": (BIAS, self.b, half_bias)}
        for label, (name, orig, new) in cases.items():
            with self.subTest(label):
                self.assertIsNone(self.refused(name, orig, new))


class Replies(unittest.TestCase):
    def test_python_block(self):
        self.assertEqual(parse_reply("Step.\n```python\nx = 1\nx\n```\n"), Python("x = 1\nx"))

    def test_forward_block(self):
        self.assertEqual(parse_reply('```forward\n{"model": "original"}\n```'), Forward('{"model": "original"}'))

    def test_no_block_is_final(self):
        self.assertEqual(parse_reply("I changed nothing."), Final("I changed nothing."))

    def test_two_blocks_are_invalid(self):
        self.assertEqual(parse_reply("```python\na\n```\n```forward\n{}\n```"), Invalid(2))

    def test_other_fence_languages_do_not_count(self):
        text = '```json\n{"a": 1}\n```\n```text\nx\n```\n```python\nprint(1)\n```'
        self.assertEqual(parse_reply(text), Python("print(1)"))
        self.assertIsInstance(parse_reply('```json\n{"a": 1}\n```'), Final)


class Requests(unittest.TestCase):
    some = sorted(IDS)[:3]

    def bad(self, obj):
        raw = obj if isinstance(obj, str) else json.dumps(obj)
        with self.assertRaises(RequestError):
            parse_request(raw, IDS)

    def test_valid(self):
        req = parse_request(json.dumps({"model": "both", "items": self.some, "texts": ["Tan Ah Kow"],
                                        "return": {"label_probs": True, "hidden_states": [0, 6], "mlp_act": [5],
                                                   "attention": [0]}}), IDS)
        self.assertEqual(req, Request("both", tuple(self.some), ("Tan Ah Kow",), True, (0, 6), (5,), (0,)))

    def test_errors(self):
        cases = {
            "not json": "{model: original}",
            "not an object": [],
            "unknown key": {"model": "original", "texts": ["a"], "batch": 2},
            "bad model": {"model": "base", "texts": ["a"]},
            "missing model": {"texts": ["a"]},
            "nothing to run": {"model": "original"},
            "too many": {"model": "original", "texts": ["a"] * 513},
            "duplicate id": {"model": "original", "items": [self.some[0]] * 2},
            "unknown id": {"model": "original", "items": ["sg-nope"]},
            "items not strings": {"model": "original", "items": [1]},
            "hidden layer out of range": {"model": "original", "texts": ["a"], "return": {"hidden_states": [7]}},
            "mlp layer out of range": {"model": "original", "texts": ["a"], "return": {"mlp_act": [6]}},
            "bool as layer": {"model": "original", "texts": ["a"], "return": {"attention": [True]}},
            "duplicate layer": {"model": "original", "texts": ["a"], "return": {"mlp_act": [1, 1]}},
            "attention on too many texts": {"model": "original", "texts": ["a"] * 65, "return": {"attention": [0]}},
            "unknown return key": {"model": "original", "texts": ["a"], "return": {"logits": True}},
            "label_probs not bool": {"model": "original", "texts": ["a"], "return": {"label_probs": 1}},
        }
        for what, obj in cases.items():
            with self.subTest(what):
                self.bad(obj)

    def test_limits_are_inclusive(self):
        parse_request(json.dumps({"model": "original", "texts": ["a"] * 512}), IDS)
        parse_request(json.dumps({"model": "original", "texts": ["a"] * 64, "return": {"attention": [5]}}), IDS)


class EditDiff(unittest.TestCase):
    def test_round_trip_matches_manifest_hashes(self):
        orig = load_file(WEIGHTS)
        with tempfile.TemporaryDirectory() as tmp:
            box, out = Box(Path(tmp).resolve()), Path(tmp, "edit")
            edited = box.edited
            edited.mkdir(parents=True)
            a = orig[MLP].copy()
            a[0, :3] += 1.0
            b = orig[GAMMA].copy()
            b[7] = -b[7]
            np.save(edited / f"{MLP}.npy", a)
            np.save(edited / f"{GAMMA}.npy", b)
            manifest = write_edit(box, WEIGHTS, out)
            rebuilt = apply_npz(out / "edit.npz", WEIGHTS)
        by_name = {t["name"]: t for t in manifest["tensors"]}
        self.assertEqual({n: t["entries_changed"] for n, t in by_name.items()}, {MLP: 3, GAMMA: 1})
        self.assertEqual(sorted(rebuilt), sorted([MLP, GAMMA]))
        for name, want in ((MLP, a), (GAMMA, b)):
            self.assertEqual(sha256(rebuilt[name]), by_name[name]["sha256_edited"])
            self.assertEqual(sha256(orig[name]), by_name[name]["sha256_original"])
            self.assertTrue(np.array_equal(rebuilt[name], want))

    def test_refused_file_blocks_the_submission(self):
        orig = load_file(WEIGHTS)
        with tempfile.TemporaryDirectory() as tmp:
            box = Box(Path(tmp).resolve())
            box.edited.mkdir(parents=True)
            np.save(box.edited / f"{MLP}.npy", 3 * orig[MLP])
            with self.assertRaises(ValueError):
                write_edit(box, WEIGHTS, Path(tmp, "edit"))


class EditFiles(unittest.TestCase):
    """read_edits runs outside the sandbox on files the editor's code controls."""

    orig = load_file(WEIGHTS)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.box = Box(self.tmp / "box")
        self.box.edited.mkdir(parents=True)
        self.secret = self.tmp / "secret.txt"
        self.secret.write_text("SECRET-EVAL-TEXT\n")
        self.good = self.orig[MLP].copy()
        self.good[0, 0] += 0.5

    def refused(self) -> dict:
        edits, refused = read_edits(self.box, self.orig)
        self.assertEqual(edits, {})
        return refused

    def test_symlink_is_refused_without_reading_its_target(self):
        os.symlink(self.secret, self.box.edited / f"{MLP}.npy")
        reason = self.refused()[MLP]
        self.assertIn("symbolic link", reason)
        self.assertNotIn("SECRET", reason)

    def test_refusal_never_quotes_file_bytes(self):
        header = b'{"SECRET-EVAL": 1}'
        (self.box.edited / f"{MLP}.npy").write_bytes(b"\x93NUMPY\x01\x00" + bytes([len(header) + 1, 0]) + header + b"\n")
        self.assertNotIn("SECRET", self.refused()[MLP])

    def test_hard_link_is_refused(self):
        np.save(self.tmp / "elsewhere.npy", self.good)
        os.link(self.tmp / "elsewhere.npy", self.box.edited / f"{MLP}.npy")
        self.assertIn("hard link", self.refused()[MLP])

    def test_unloadable_files_are_refusals_not_crashes(self):
        with open(self.box.edited / f"{MLP}.npy", "wb") as f:
            np.savez(f, x=self.good)
        np.save(self.box.edited / f"{GAMMA}.npy", np.zeros(self.orig[GAMMA].shape, dtype=[("a", "<f4")]))
        refused = self.refused()
        self.assertEqual(sorted(refused), sorted([MLP, GAMMA]))
        self.assertTrue(all("not a .npy file holding one numeric array" in r for r in refused.values()))

    def test_fifo_is_refused_without_blocking(self):
        os.mkfifo(self.box.edited / f"{MLP}.npy")
        out = []
        t = threading.Thread(target=lambda: out.append(self.refused()), daemon=True)
        t.start()
        t.join(10)
        self.assertFalse(t.is_alive(), "read_edits blocked on a FIFO")
        self.assertIn("not a regular file", out[0][MLP])

    def test_symlinked_folders_are_refused(self):
        elsewhere = self.tmp / "elsewhere"
        (elsewhere / "edited").mkdir(parents=True)
        np.save(elsewhere / "edited" / f"{MLP}.npy", self.good)
        _rmtree(self.box.edited)
        os.symlink(elsewhere / "edited", self.box.edited)
        with self.assertRaises(ValueError):
            read_edits(self.box, self.orig)
        _rmtree(self.box.work)
        os.symlink(elsewhere, self.box.work)
        with self.assertRaises(ValueError):
            read_edits(self.box, self.orig)

    def test_regular_file_is_read(self):
        np.save(self.box.edited / f"{MLP}.npy", self.good)
        edits, refused = read_edits(self.box, self.orig)
        self.assertEqual(refused, {})
        self.assertTrue(np.array_equal(edits[MLP], self.good))


class ServerWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = setup_box("unittest-server", fresh=True)
        cls.server = Server(cls.box, PRACTICE)
        cls.orig = load_file(WEIGHTS)

    def params(self):
        return {n: p.detach().numpy().copy() for n, p in self.server.model.named_parameters()}

    def test_gamma_beta_names_map_to_weight_bias(self):
        self.assertEqual(hf_name(GAMMA), "bert.encoder.layer.2.output.LayerNorm.weight")
        self.assertEqual(hf_name(GAMMA.replace("gamma", "beta")), "bert.encoder.layer.2.output.LayerNorm.bias")
        self.assertEqual(hf_name(MLP), MLP)

    def test_zero_edits_give_bitwise_original_params(self):
        self.server.set_weights({GAMMA: self.orig[GAMMA] + 1})
        self.server.set_weights({})
        params = self.params()
        self.assertEqual(sorted(params), sorted(hf_name(n) for n in self.orig))
        for name, a in self.orig.items():
            self.assertTrue(np.array_equal(params[hf_name(name)], a), name)

    def test_gamma_edit_changes_exactly_that_param(self):
        new = self.orig[GAMMA].copy()
        new[5] += 0.25
        self.server.set_weights({GAMMA: new})
        params = self.params()
        self.server.set_weights({})
        changed = [hf_name(n) for n, a in self.orig.items() if not np.array_equal(params[hf_name(n)], a)]
        self.assertEqual(changed, [hf_name(GAMMA)])
        self.assertTrue(np.array_equal(params[hf_name(GAMMA)], new))

    def test_edit_reaches_the_edited_run_and_is_undone_after(self):
        name = "bert.encoder.layer.5.output.dense.weight"
        noisy = self.orig[name] + np.random.default_rng(0).normal(0, 0.5, self.orig[name].shape).astype(np.float32)
        items = sorted(IDS)[:20]
        req = json.dumps({"model": "both", "items": items})
        before = self.server.handle(1, self.server.parse(json.dumps({"model": "original", "items": items}))).summary
        np.save(self.box.edited / f"{name}.npy", noisy)
        try:
            both = self.server.handle(2, self.server.parse(req)).summary
        finally:
            (self.box.edited / f"{name}.npy").unlink()
        after = self.server.handle(3, self.server.parse(json.dumps({"model": "original", "items": items}))).summary
        rows = [json.loads(l) for l in (self.box.work / "pred_2.jsonl").read_text().splitlines()]
        preds = {m: [r["pred"] for r in rows if r["model"] == m] for m in ("original", "edited")}
        self.assertEqual(both["edited_tensors"], [name])
        self.assertNotEqual(preds["original"], preds["edited"])
        self.assertNotEqual(both["diff"]["f1_edited_minus_original"], 0.0)
        self.assertEqual(both["original"], before["original"])
        self.assertEqual(after["original"], before["original"])

    def test_file_written_around_save_edit_is_refused(self):
        np.save(self.box.edited / f"{MLP}.npy", 2 * self.orig[MLP])
        try:
            with self.assertRaises(RequestError):
                self.server.handle(1, self.server.parse(json.dumps({"model": "edited", "texts": ["a"]})))
        finally:
            (self.box.edited / f"{MLP}.npy").unlink()

    def test_symlinks_in_work_do_not_redirect_the_servers_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, folder = Path(tmp, "target.txt"), Path(tmp, "folder")
            target.write_text("ORIGINAL\n")
            folder.mkdir()
            pred, arrays = self.box.work / "pred_7.jsonl", self.box.work / "fwd_7"
            os.symlink(target, pred)
            os.symlink(folder, arrays)
            req = {"model": "original", "texts": ["Tan Ah Kow"], "return": {"label_probs": True}}
            result = self.server.handle(7, self.server.parse(json.dumps(req)))
            self.assertEqual(target.read_text(), "ORIGINAL\n")
            self.assertEqual(list(folder.iterdir()), [])
        self.assertFalse(pred.is_symlink() or arrays.is_symlink())
        self.assertEqual(pred.read_text(), result.pred)
        self.assertEqual(json.loads((arrays / "meta.json").read_text()), result.meta)
        self.assertEqual(sorted(p.name for p in arrays.iterdir()), ["label_probs.npy", "meta.json"])
        self.assertEqual(list(self.box.staging.iterdir()), [])

    def test_symlinked_work_is_refused(self):
        real = self.box.root / "work-real"
        self.box.work.rename(real)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.symlink(tmp, self.box.work)
                for model in ("original", "edited"):
                    with self.subTest(model), self.assertRaises(RequestError):
                        self.server.handle(8, self.server.parse(json.dumps({"model": model, "texts": ["a"]})))
                self.assertEqual(list(Path(tmp).iterdir()), [])
        finally:
            self.box.work.unlink()
            real.rename(self.box.work)


class BoxSetup(unittest.TestCase):
    def test_setup_does_not_follow_a_symlinked_work_folder(self):
        box = setup_box("unittest-setup", fresh=True)
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp, "keep.txt")
            outside.write_text("x")
            outside.chmod(0o600)
            _rmtree(box.work)
            os.symlink(tmp, box.work)
            box = setup_box("unittest-setup", fresh=True)
            self.assertEqual((outside.read_text(), stat.S_IMODE(outside.stat().st_mode)), ("x", 0o600))
        self.assertTrue(box.work.is_dir() and not box.work.is_symlink())
        self.assertTrue(box.edited.is_dir())


def guarded(action: str) -> str:
    """Kernel code that prints DENIED only when the sandbox refuses the action."""
    return f"try:\n{textwrap.indent(action, '    ')}\n    print('ALLOWED')\nexcept PermissionError:\n    print('DENIED')"


class KernelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.box = setup_box("unittest-kernel", fresh=True)
        cls.kernel = Kernel(cls.box, Path(cls.tmp.name, "kernel.log"))
        cls.kernel.start()

    @classmethod
    def tearDownClass(cls):
        cls.kernel.close()
        cls.tmp.cleanup()

    def out(self, code: str) -> str:
        return self.kernel.run(code).output.strip()


class Notebook(KernelCase):
    def test_variables_persist(self):
        self.out("x = 41")
        self.assertEqual(self.out("x + 1"), "42")

    def test_last_expression_is_displayed(self):
        self.assertEqual(self.out("print('a')\n'b'"), "a\n'b'")
        self.assertEqual(self.out("y = 3"), "")
        self.assertEqual(self.out("None"), "")

    def test_traceback_in_output(self):
        out = self.out("z = 1\n1 / 0")
        self.assertIn("ZeroDivisionError", out)
        self.assertIn("<cell", out)
        self.assertEqual(self.out("z"), "1")

    def test_output_cut_at_10000_characters(self):
        out = self.kernel.run("print('a' * 25000)").output
        self.assertEqual(out[:10_000], "a" * 10_000)
        self.assertEqual(out[10_000:], "\n[output cut at 10,000 of 25,001 characters]")

    def test_predefined_names(self):
        self.assertEqual(self.out("len(NAMES), NAMES[0], edited(), tok.encode('Tan').tokens"),
                         "(103, 'bert.embeddings.LayerNorm.beta', [], ['[CLS]', 'tan', '[SEP]'])")
        self.assertEqual(self.out("import torch; torch.is_grad_enabled()"),
                         "False")

    def test_save_edit_refuses_and_accepts(self):
        self.assertIn("embeddings", self.out("save_edit(NAMES[2], load(NAMES[2]) + 1)"))
        self.assertIn("classification head", self.out("save_edit('classifier.bias', load('classifier.bias') + 1)"))
        self.assertIn("c=1.5", self.out(f"save_edit('{MLP}', 1.5 * load('{MLP}'))"))
        self.assertEqual(self.out(f"w = load('{MLP}'); w[0, 0] += 0.5; save_edit('{MLP}', w); edited()"), f"['{MLP}']")
        self.assertEqual(self.out(f"float(load('{MLP}')[0, 0] - load_original('{MLP}')[0, 0])"), "0.5")
        self.assertEqual(self.out(f"drop_edit('{MLP}'); edited()"), "[]")


class Sandbox(KernelCase):
    def test_project_files_are_unreadable(self):
        for path in (SETS / "sg_eval.jsonl", ROOT / "docs" / "weight-space-pii-handoff.md",
                     Path("/System/Volumes/Data") / str(SETS / "sg_eval.jsonl").lstrip("/")):
            with self.subTest(str(path)):
                self.assertEqual(self.out(guarded(f"open({str(path)!r}).read(10)")), "DENIED")

    def test_subprocess_inherits_the_sandbox(self):
        out = self.out(f"import subprocess; r = subprocess.run(['cat', {str(SETS / 'sg_eval.jsonl')!r}], "
                       "capture_output=True, text=True); (r.returncode, r.stdout[:20], r.stderr)")
        self.assertIn("Operation not permitted", out)
        self.assertTrue(out.startswith("(1, ''"), out)

    def test_hard_link_of_a_protected_file_fails(self):
        for src in (SETS / "sg_eval.jsonl", self.box.practice_file):
            with self.subTest(str(src)):
                self.assertEqual(self.out(guarded(f"import os; os.link({str(src)!r}, 'linked')")), "DENIED")

    def test_cpu_workload_runs(self):
        code = ("import numpy as np, torch, sklearn.linear_model\na = torch.randn(64, 64)\nnp.save('w.npy', (a @ a).numpy())\n"
                "x = np.random.rand(40, 3)\nsklearn.linear_model.LogisticRegression().fit(x, x[:, 0] > .5)\n"
                "np.load('w.npy').shape, tok.encode('Tan Ah Kow').tokens[1]")
        self.assertEqual(self.out(code), "((64, 64), 'tan')")

    def test_writes_outside_work_fail(self):
        for path in (self.box.model / "x", self.box.root / "x", ROOT / "x", Path("/private/tmp/wsp-x")):
            with self.subTest(str(path)):
                self.assertEqual(self.out(guarded(f"open({str(path)!r}, 'w').write('x')")), "DENIED")
        self.assertEqual(self.out(guarded("open('scratch.txt', 'w').write('x')")), "ALLOWED")

    def test_network_connect_fails(self):
        for host, port in (("1.1.1.1", 80), ("127.0.0.1", 22)):
            with self.subTest(host):
                self.assertEqual(self.out(guarded(f"import socket; socket.create_connection(({host!r}, {port}), "
                                                  "timeout=5)")), "DENIED")

    def test_system_services_cannot_run_work_outside_the_sandbox(self):
        label, marker = "wsp.test.escape", self.box.root / "escaped"
        try:
            self.kernel.run(f"import subprocess; subprocess.run(['/bin/launchctl', 'submit', '-l', {label!r}, '--', "
                            f"'/usr/bin/touch', {str(marker)!r}])")
            time.sleep(2)
            listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
            self.assertNotIn(label, listed)
            self.assertFalse(marker.exists())
        finally:
            subprocess.run(["launchctl", "remove", label], capture_output=True)
            marker.unlink(missing_ok=True)
        # "get name" is answered without an Apple Event, so ask for something only a delivered event returns.
        for argv in (["/usr/bin/osascript", "-e", 'tell application "Finder" to get name of every disk'],
                     ["/usr/bin/open", "-g", "-a", "Calculator"]):
            with self.subTest(argv[0]):
                out = self.out(f"import subprocess; subprocess.run({argv!r}, capture_output=True).returncode")
                self.assertNotEqual(out, "0")

    def test_model_and_practice_are_readable(self):
        self.assertEqual(self.out(f"import json; json.load(open({str(self.box.model / 'config.json')!r}))['hidden_size']"),
                         "384")
        self.assertEqual(self.out(f"sum(1 for _ in open({str(self.box.practice_file)!r}))"), str(len(PRACTICE)))

    def test_torch_imports_and_the_gpu_is_off(self):
        self.assertEqual(self.out("import torch, numpy, scipy, sklearn, pandas; torch.__version__"), repr(torch.__version__))
        self.assertEqual(self.out("torch.backends.mps.is_available()"), "False")
        self.assertIn("RuntimeError", self.out("torch.ones(1, device='mps')"))


class KernelLimits(unittest.TestCase):
    def test_timeout_and_death_restart_the_kernel(self):
        box = setup_box("unittest-limit", fresh=True)
        with tempfile.TemporaryDirectory() as tmp:
            k = Kernel(box, Path(tmp, "kernel.log"), timeout=3)
            k.start()
            try:
                k.run("v = 1")
                r = k.run("import time; time.sleep(60)")
                self.assertTrue(r.restarted)
                self.assertIn("stopped after", r.output)
                self.assertIn("restarted", r.output)
                self.assertIn("NameError", k.run("v").output)
                r = k.run("import os; os._exit(1)")
                self.assertTrue(r.restarted)
                self.assertIn("died", r.output)
                self.assertEqual(k.run("1 + 1").output.strip(), "2")
            finally:
                k.close()


class KernelProtocol(unittest.TestCase):
    """Cell code can write to the kernel's reply pipe; that must cost a restart, never a desync or a crash."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kernel = Kernel(setup_box("unittest-proto", fresh=True), Path(cls.tmp.name, "kernel.log"), timeout=60)
        cls.kernel.start()

    @classmethod
    def tearDownClass(cls):
        cls.kernel.close()
        cls.tmp.cleanup()

    def test_protocol_fds_are_not_in_argv(self):
        self.assertEqual(self.kernel.run("import sys; len(sys.argv)").output.strip(), "1")

    def test_forged_reply_lines_restart_the_kernel(self):
        forgeries = {"fake reply": b'{"output": "FAKE", "total": 4, "seconds": 0}\n',
                     "missing keys": b'{"output": "x"}\n', "partial line": b'{"output"', "not json": b"oops\n",
                     "not an object": b"[1]\n", "bad utf-8": b"\xff\n"}
        for what, line in forgeries.items():
            with self.subTest(what):
                code = (f"import os, stat\nfor fd in range(3, 256):\n    try:\n"
                        f"        if stat.S_ISFIFO(os.fstat(fd).st_mode): os.write(fd, {line!r})\n"
                        "    except OSError: pass\nprint('REAL 1')")
                r = self.kernel.run(code)
                self.assertTrue(r.restarted, r.output)
                self.assertIn("garbled", r.output)
                self.assertEqual(self.kernel.run("print('REAL 2')").output.strip(), "REAL 2")


class FakeEditor:
    def __init__(self, replies):
        self.replies, self.messages = list(replies), []
        self.session_id, self.retries = "fake-session", []

    def send(self, message):
        self.messages.append(message)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return EditorReply(reply, {"input_tokens": 10, "output_tokens": 5}, 0.01, 0.01 * len(self.messages),
                           self.session_id, 0.0)

    def info(self):
        return {"model": "fake", "effort": "max", "session_id": self.session_id}


def block(lang, body):
    return f"Next step.\n```{lang}\n{body}\n```"


class Driver(unittest.TestCase):
    ids = sorted(IDS)[:3]

    def run_fake(self, replies, budget, max_turns=None, ask_context=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = FakeEditor(replies)
        result = run("unittest-driver", budget, max_turns, ask_context=ask_context,
                     make_editor=lambda box, log: fake, runs_root=Path(tmp.name))
        rundir = Path(tmp.name, "unittest-driver")
        events = [json.loads(l) for l in (rundir / "transcript.jsonl").read_text().splitlines()]
        return fake, result, rundir, events

    def test_full_session(self):
        replies = [
            block("python", "x = 41\nx + 1"),
            block("forward", json.dumps({"model": "original", "items": ["sg-nope"]})),
            "Thinking out loud, no block.",
            block("python", "1") + "\n" + block("forward", "{}"),
            block("python", f"w = load('{MLP}'); w[0, 0] += 0.5; save_edit('{MLP}', w); edited()"),
            block("forward", json.dumps({"model": "both", "items": self.ids})),
            block("forward", json.dumps({"model": "edited", "texts": ["Tan Ah Kow, S1234567D"],
                                         "return": {"label_probs": True}})),
            "FINAL: I raised one weight.",
            "I have no tools.",
        ]
        fake, result, rundir, events = self.run_fake(replies, budget=2, max_turns=10, ask_context=True)
        m = fake.messages
        self.assertIn("TASK.", m[0])
        self.assertIn("OUTPUT of Python run 1 of at most 300:\n```\n42\n```", m[1])
        self.assertTrue(m[2].startswith("Forward request failed and was not counted"))
        self.assertIn("Forward passes left: 2.", m[2])
        self.assertIn("no ```python or ```forward block, so nothing was run. 2 forward passes are left", m[3])
        self.assertIn("Your reply had 2 code blocks", m[4])
        self.assertIn(f"['{MLP}']", m[5])
        self.assertTrue(m[6].startswith("Forward pass 1 of 2 done"))
        self.assertIn('"f1_edited_minus_original"', m[6])
        self.assertIn("Forward passes left: 1.", m[6])
        self.assertNotIn("All 2 forward passes are used", m[6])
        self.assertTrue(m[7].startswith("Forward pass 2 of 2 done"))
        self.assertIn("All 2 forward passes are used. Make your final choice", m[7])
        self.assertIn("Report only what you can see", m[8])
        self.assertEqual(len(m), 9)

        self.assertEqual({k: result[k] for k in ("turns", "python_runs", "passes_used", "failed_forwards",
                                                  "invalid_replies", "ended_by", "context_check")},
                         {"turns": 8, "python_runs": 2, "passes_used": 2, "failed_forwards": 1,
                          "invalid_replies": 1, "ended_by": "final_answer", "context_check": True})
        self.assertEqual(result["usage"]["input_tokens"], 90)
        self.assertEqual(result["cost_usd"], 0.09)
        self.assertEqual(json.loads((rundir / "run.json").read_text())["passes_used"], 2)
        self.assertEqual((rundir / "final_answer.md").read_text(), "FINAL: I raised one weight.\n")
        self.assertIn("I have no tools.", (rundir / "context_check.md").read_text())

        manifest = json.loads((rundir / "edit" / "manifest.json").read_text())
        self.assertEqual([(t["name"], t["entries_changed"]) for t in manifest["tensors"]], [(MLP, 1)])
        self.assertEqual(sha256(apply_npz(rundir / "edit" / "edit.npz", WEIGHTS)[MLP]),
                         manifest["tensors"][0]["sha256_edited"])
        self.assertEqual(sorted(p.name for p in (rundir / "forward" / "1").iterdir()),
                         ["edit.npz", "edit_manifest.json", "pred.jsonl", "request.json", "summary.json"])
        self.assertEqual(sorted(p.name for p in (rundir / "forward" / "2").iterdir()),
                         ["edit_manifest.json", "meta.json", "pred.jsonl", "request.json", "summary.json"])
        for k in ("1", "2"):
            ran = json.loads((rundir / "forward" / k / "edit_manifest.json").read_text())
            self.assertEqual((ran["tensors"], ran["npz"]), (manifest["tensors"], "1/edit.npz"))
        self.assertEqual(sha256(apply_npz(rundir / "forward" / "1" / "edit.npz", WEIGHTS)[MLP]),
                         manifest["tensors"][0]["sha256_edited"])
        self.assertFalse((rundir / ".edit_snapshot").exists())
        saved = json.loads((rundir / "run.json").read_text())
        self.assertEqual(saved["lock_sha256"], sha256_file(ROOT / "lock.json"))
        self.assertFalse(saved["cli_session_copied"])
        self.assertEqual([e["kind"] for e in events if e["kind"] != "user"],
                         ["assistant", "python", "assistant", "forward_failed", "assistant", "assistant", "invalid",
                          "assistant", "python", "assistant", "forward", "assistant", "forward", "assistant",
                          "context_check"])
        self.assertTrue((rundir / "kernel.log").exists())

    def test_turn_limit_forces_the_final_answer(self):
        replies = [block("python", "1"), block("python", "2"), block("python", "3")]
        fake, result, rundir, _ = self.run_fake(replies, budget=5, max_turns=3)
        self.assertTrue(fake.messages[2].startswith("OUTPUT of Python run 2 of at most 300"))
        self.assertIn("The turn limit for this session is reached. Make your final choice", fake.messages[2])
        self.assertEqual((result["turns"], result["python_runs"], result["ended_by"]), (3, 2, "turn_limit"))
        self.assertEqual((rundir / "final_answer.md").read_text(), replies[2] + "\n")
        self.assertEqual(json.loads((rundir / "edit" / "manifest.json").read_text())["n_tensors"], 0)

    def test_no_block_at_the_turn_limit_gets_only_the_wrap_up(self):
        fake, result, _, _ = self.run_fake(["no block", "done"], budget=5, max_turns=2)
        self.assertTrue(fake.messages[1].startswith("The turn limit for this session is reached."))
        self.assertEqual(result["ended_by"], "turn_limit")

    def test_last_pass_at_the_turn_limit_asks_for_the_final_choice_once(self):
        replies = [block("python", "1"), block("forward", json.dumps({"model": "original", "texts": ["a"]})), "done"]
        fake, result, _, _ = self.run_fake(replies, budget=1, max_turns=3)
        self.assertEqual(fake.messages[2].count("Make your final choice"), 1)
        self.assertIn("The turn limit for this session is reached.", fake.messages[2])
        self.assertEqual(result["ended_by"], "turn_limit")

    def test_no_turn_limit(self):
        replies = [block("python", "1")] * 12 + [block("forward", json.dumps({"model": "original", "texts": ["a"]})),
                                                 "done"]
        fake, result, _, _ = self.run_fake(replies, budget=1)
        self.assertEqual((result["turns"], result["max_turns"], result["ended_by"]), (14, None, "final_answer"))

    def test_internal_server_error_is_a_failed_forward(self):
        replies = [block("forward", '{"model": "original", "texts": ["\\ud800"]}'),
                   block("forward", json.dumps({"model": "original", "texts": ["a"]})), "done"]
        fake, result, _, events = self.run_fake(replies, budget=1)
        self.assertIn("internal error (TypeError) and was not counted", fake.messages[1])
        self.assertIn("Forward passes left: 1.", fake.messages[1])
        self.assertEqual((result["failed_forwards"], result["passes_used"]), (1, 1))
        errors = [e for e in events if e["kind"] == "forward_error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Traceback", errors[0]["traceback"])

    def test_unloadable_edit_is_an_edit_error_in_run_json(self):
        code = f"import numpy as np\nwith open('edited/{MLP}.npy', 'wb') as f: np.savez(f, x=np.ones(3))"
        _, result, rundir, _ = self.run_fake([block("python", code), "done"], budget=5, max_turns=2)
        self.assertIn("not a .npy file holding one numeric array", result["edit_error"])
        self.assertIsNone(result["edit"])
        self.assertEqual(json.loads((rundir / "run.json").read_text())["edit_error"], result["edit_error"])

    def test_any_write_edit_failure_is_an_edit_error(self):
        with mock.patch("wsp.harness.driver.write_edit", side_effect=OSError("disk full")):
            _, result, rundir, _ = self.run_fake(["no block", "done"], budget=5, max_turns=2)
        self.assertEqual(result["edit_error"], "OSError: disk full")
        self.assertTrue((rundir / "run.json").exists())

    def test_each_pass_records_the_edit_it_ran(self):
        save = f"w = load('{GAMMA}'); w[0] += {{}}; save_edit('{GAMMA}', w)"
        texts = json.dumps({"model": "edited", "texts": ["a"]})
        replies = [block("forward", json.dumps({"model": "original", "texts": ["a"]})),
                   block("python", save.format(0.5)), block("forward", texts),
                   block("python", save.format(0.25)), block("forward", texts), "done"]
        _, result, rundir, _ = self.run_fake(replies, budget=3)
        ran = [json.loads((rundir / "forward" / k / "edit_manifest.json").read_text()) for k in ("1", "2", "3")]
        self.assertEqual((ran[0]["n_tensors"], ran[0]["npz"]), (0, None))
        self.assertEqual([r["npz"] for r in ran[1:]], ["2/edit.npz", "3/edit.npz"])
        for k, r in zip(("2", "3"), ran[1:], strict=True):
            rebuilt = apply_npz(rundir / "forward" / k / "edit.npz", WEIGHTS)[GAMMA]
            self.assertEqual(sha256(rebuilt), r["tensors"][0]["sha256_edited"])
        self.assertNotEqual(ran[1]["tensors"], ran[2]["tensors"])
        self.assertEqual(result["passes_used"], 3)

    def test_small_work_files_are_kept_and_nothing_else(self):
        code = textwrap.dedent("""\
            import os
            os.makedirs('sub', exist_ok=True); os.makedirs('fwd_9', exist_ok=True)
            open('notes.md', 'w').write('finding: 42')
            open('sub/plan.txt', 'w').write('plan')
            open('big.bin', 'wb').write(b'x' * (2 << 20))
            open('pred_1.jsonl', 'w').write('{}')
            open('fwd_9/a.npy', 'wb').write(b'arr')
            os.symlink('/etc/hosts', 'hosts_link')
            open('linked.txt', 'w').write('x'); os.link('linked.txt', 'linked_again.txt')
            'ok'""")
        _, result, rundir, _ = self.run_fake([block("python", code), "done"], budget=5, max_turns=2)
        notes = rundir / "work_notes"
        kept = sorted(str(p.relative_to(notes)) for p in notes.rglob("*") if p.is_file())
        self.assertEqual(kept, ["notes.md", "sub/plan.txt"])
        self.assertEqual((notes / "notes.md").read_text(), "finding: 42")
        self.assertEqual(sorted(result["work_notes"]["skipped"]),
                         [["big.bin", f"{2 << 20} bytes"], ["hosts_link", "symlink"],
                          ["linked.txt", "not a plain file"], ["linked_again.txt", "not a plain file"]])

    def test_the_cli_session_file_is_copied(self):
        with tempfile.TemporaryDirectory() as config:
            session = Path(config, "projects", "some-cwd", "fake-session.jsonl")
            session.parent.mkdir(parents=True)
            session.write_text('{"type": "assistant"}\n')
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": config}):
                _, result, rundir, _ = self.run_fake(["no block", "done"], budget=5, max_turns=2)
            self.assertTrue(result["cli_session_copied"])
            self.assertEqual((rundir / "cli_session.jsonl").read_text(), session.read_text())

    def test_run_files_are_written_when_the_session_fails(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = FakeEditor([block("python", "1"), RuntimeError("the editor CLI kept failing")])
        with self.assertRaises(RuntimeError):
            run("unittest-driver", 2, make_editor=lambda box, log: fake, runs_root=Path(tmp.name))
        rundir = Path(tmp.name, "unittest-driver")
        result = json.loads((rundir / "run.json").read_text())
        self.assertEqual((result["ended_by"], result["turns"], result["python_runs"]), ("error", 1, 1))
        self.assertIn("kept failing", result["error"])
        self.assertIn("No final answer", (rundir / "final_answer.md").read_text())
        self.assertEqual(result["edit"], {"n_tensors": 0, "entries_changed": 0})
        events = [json.loads(l) for l in (rundir / "transcript.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["kind"], "error")


FAKE_CLI = """#!/bin/sh
n=$(cat "$STATE" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$STATE"
echo "$@" >> "$ARGS"; cat >> "$STDIN"; echo >> "$STDIN"
if [ "$n" -le "$FAILS" ]; then echo '{"type":"result","subtype":"error","is_error":true,"result":"usage limit","total_cost_usd":0.'$n',"usage":{"output_tokens":1}}'; exit 1; fi
echo '{"type":"result","subtype":"success","is_error":false,"result":"reply '$n'","session_id":"s","total_cost_usd":0.'$n',"usage":{"output_tokens":3}}'
"""


class EditorCli(unittest.TestCase):
    def test_retries_on_the_same_session_and_resumes_after_the_first_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            cli = t / "claude"
            cli.write_text(FAKE_CLI)
            cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
            env = {"PATH": f"{t}:{os.environ['PATH']}", "STATE": str(t / "n"), "ARGS": str(t / "args"),
                   "STDIN": str(t / "stdin"), "FAILS": "2", "CLAUDE_CONFIG_DIR": str(t / "config")}
            logged, slept = [], []
            with mock.patch.dict(os.environ, env):
                ed = Editor(t, lambda kind, **f: logged.append(kind), sleep=slept.append)
                first = ed.send("hello")
                second = ed.send("again")
            args = (t / "args").read_text().splitlines()
            stdin = (t / "stdin").read_text().splitlines()
        self.assertEqual((first.text, second.text), ("reply 3", "reply 4"))
        self.assertEqual((first.cost, first.total_cost, second.total_cost), (0.3, 0.3, 0.4))
        self.assertAlmostEqual(second.cost, 0.1)
        self.assertEqual(slept, [60, 120])
        self.assertEqual([(r["wait"], r["total_cost_usd"], r["usage"]) for r in ed.retries],
                         [(60, 0.1, {"output_tokens": 1}), (120, 0.2, {"output_tokens": 1})])
        self.assertEqual(logged, ["cli_retry", "cli_retry"])
        self.assertTrue(all(f"--session-id {ed.session_id}" in a for a in args[:3]))
        self.assertIn(f"--resume {ed.session_id}", args[3])
        self.assertIn("--tools  --strict-mcp-config", args[0])
        self.assertIn("--setting-sources  --system-prompt Answer the user's request.", args[0])
        self.assertEqual(stdin, ["hello", "hello", "hello", "again"])


if __name__ == "__main__":
    unittest.main()

"""The forward-pass server: the only way the editor runs the model.

It predicts spans with `wsp.predict.predict` and grades them with `wsp.score`, so a forward pass
scores exactly as the frozen M5 and M7 evaluation does. Requested activations come from a second,
hooked pass per text.
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.lib.format import open_memmap
from safetensors.numpy import load_file

from wsp.data import MAX_LEN, Doc, Span
from wsp.harness.box import Box, _rmtree, open_dir
from wsp.harness.edit import read_edits
from wsp.predict import load, predict
from wsp.score import FAILURES, SG_LABELS, Grade, bootstrap, grade, summarize

MAX_TEXTS = 512
MAX_ATTENTION_TEXTS = 64
MODELS = ("original", "edited", "both")
RETURN_KEYS = ("label_probs", "hidden_states", "mlp_act", "attention")
HF_SUFFIX = {"LayerNorm.gamma": "LayerNorm.weight", "LayerNorm.beta": "LayerNorm.bias"}


class RequestError(ValueError):
    """A forward request the server will not run. It costs no forward pass."""


@dataclass(frozen=True)
class Request:
    model: Literal["original", "edited", "both"]
    items: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    label_probs: bool = False
    hidden_states: tuple[int, ...] = ()
    mlp_act: tuple[int, ...] = ()
    attention: tuple[int, ...] = ()

    @property
    def wants_arrays(self) -> bool:
        return bool(self.label_probs or self.hidden_states or self.mlp_act or self.attention)


def _str_list(obj: dict, key: str) -> tuple[str, ...]:
    v = obj.get(key, [])
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise RequestError(f'"{key}" must be a list of strings')
    return tuple(v)


def _layers(ret: dict, key: str, top: int) -> tuple[int, ...]:
    v = ret.get(key, [])
    if not isinstance(v, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in v):
        raise RequestError(f'"return.{key}" must be a list of layer indices')
    if bad := [i for i in v if not 0 <= i <= top]:
        raise RequestError(f'"return.{key}" indices must be in 0..{top}; got {bad}')
    if len(set(v)) != len(v):
        raise RequestError(f'"return.{key}" lists a layer twice')
    return tuple(v)


@dataclass(frozen=True)
class Result:
    """What one forward pass returns. `pred` and `meta` are the contents of the files it put in work/."""
    summary: dict
    pred: str
    meta: dict | None


def parse_request(raw: str, practice_ids, n_layers: int = 6) -> Request:
    """Parse the body of a ```forward block. Raises RequestError for anything invalid."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RequestError(f"the block is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise RequestError("the request must be a JSON object")
    if unknown := sorted(set(obj) - {"model", "items", "texts", "return"}):
        raise RequestError(f"unknown keys: {unknown}")
    if obj.get("model") not in MODELS:
        raise RequestError(f'"model" must be one of {list(MODELS)}')
    items, texts = _str_list(obj, "items"), _str_list(obj, "texts")
    if not 1 <= len(items) + len(texts) <= MAX_TEXTS:
        raise RequestError(f"items and texts together must number 1 to {MAX_TEXTS}; got {len(items) + len(texts)}")
    if len(set(items)) != len(items):
        raise RequestError("an item id is listed twice")
    if missing := [i for i in items if i not in practice_ids]:
        raise RequestError(f"unknown practice ids: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    ret = obj.get("return", {})
    if not isinstance(ret, dict):
        raise RequestError('"return" must be a JSON object')
    if unknown := sorted(set(ret) - set(RETURN_KEYS)):
        raise RequestError(f'unknown "return" keys: {unknown}')
    if not isinstance(ret.get("label_probs", False), bool):
        raise RequestError('"return.label_probs" must be true or false')
    req = Request(obj["model"], items, texts, ret.get("label_probs", False),
                  _layers(ret, "hidden_states", n_layers), _layers(ret, "mlp_act", n_layers - 1),
                  _layers(ret, "attention", n_layers - 1))
    if req.attention and len(items) + len(texts) > MAX_ATTENTION_TEXTS:
        raise RequestError(f"attention can be returned for at most {MAX_ATTENTION_TEXTS} items and texts")
    return req


def hf_name(name: str) -> str:
    """The HF module parameter that a safetensors tensor name loads into."""
    for old, new in HF_SUFFIX.items():
        if name.endswith(old):
            return name[: -len(old)] + new
    return name


def _grade_json(g: Grade) -> dict:
    return {"by_label": {lab: {"tp": t.tp, "fp": t.fp, "fn": t.fn} for lab, t in g.by_label.items()},
            "failures": {k: g.failures[k] for k in FAILURES}}


def _span_json(text: str, s: Span) -> dict:
    return {"start": s.start, "end": s.end, "label": s.label, "text": text[s.start:s.end]}


class Server:
    def __init__(self, box: Box, practice: list[Doc]):
        self.box = box
        self.practice = {d.id: d for d in practice}
        self.tok, self.model = load(box.model)
        self.n_layers = self.model.config.num_hidden_layers
        self.params = dict(self.model.named_parameters())
        self.original = {k: v.detach().clone() for k, v in self.params.items()}
        self.original_np = load_file(box.weights)

    def parse(self, raw: str) -> Request:
        return parse_request(raw, self.practice, self.n_layers)

    def load_edits(self) -> dict[str, np.ndarray]:
        """The files in work/edited, re-checked here because editor code can write them without save_edit."""
        try:
            edits, refused = read_edits(self.box, self.original_np)
        except ValueError as e:
            raise RequestError(str(e)) from None
        if refused:
            lines = "; ".join(f"{n}: {r}" for n, r in refused.items())
            raise RequestError(f"work/edited holds edits that save_edit would refuse ({lines}). "
                               "Fix or delete them with drop_edit")
        return edits

    def set_weights(self, edits: dict[str, np.ndarray]) -> None:
        mapped = {hf_name(n): torch.from_numpy(a) for n, a in edits.items()}
        for name, p in self.params.items():
            p.data.copy_(mapped.get(name, self.original[name]))

    def handle(self, k: int, req: Request) -> Result:
        start = time.monotonic()
        edits = self.load_edits() if req.model != "original" else {}
        models = ("original", "edited") if req.model == "both" else (req.model,)
        docs = [self.practice[i] for i in req.items]
        texts = [d.text for d in docs] + list(req.texts)
        ids = [*req.items, *([None] * len(req.texts))]
        pred_name, arrays_name = f"pred_{k}.jsonl", f"fwd_{k}"
        stage = Path(tempfile.mkdtemp(dir=self.box.staging))
        summary: dict = {}
        grades: dict[str, list[Grade]] = {}
        shapes: dict[str, list[int]] = {}
        meta: dict | None = None
        rows: list[str] = []
        try:
            try:
                for m in models:
                    self.set_weights(edits if m == "edited" else {})
                    spans = predict(self.tok, self.model, texts)
                    grades[m] = [grade(d, s, SG_LABELS) for d, s in zip(docs, spans)]
                    for i, (text, s) in enumerate(zip(texts, spans)):
                        row = {"model": m, "index": i, "id": ids[i], "pred": [_span_json(text, x) for x in s]}
                        if i < len(docs):
                            row["grade"] = _grade_json(grades[m][i])
                        rows.append(json.dumps(row, ensure_ascii=False) + "\n")
                    if docs:
                        summary[m] = summarize(grades[m])
                    if req.wants_arrays:
                        prefix = f"{m}_" if req.model == "both" else ""
                        meta = self._capture(req, texts, ids, stage / arrays_name, prefix, shapes)
            finally:
                self.set_weights({})
            pred = "".join(rows)
            (stage / pred_name).write_text(pred, encoding="utf-8")
            if meta is not None:
                meta["arrays"] = shapes
                (stage / arrays_name / "meta.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n",
                                                               encoding="utf-8")
            self._publish(stage, [pred_name, *([arrays_name] if meta is not None else [])])
        finally:
            _rmtree(stage)
        summary["items"], summary["texts"] = len(docs), len(req.texts)
        if req.model == "both" and docs:
            b = bootstrap(grades["original"], grades["edited"])
            f1 = [(o.total.f1, e.total.f1) for o, e in zip(grades["original"], grades["edited"])]
            summary["diff"] = {"f1_edited_minus_original": round(b["diff"], 4), "se": round(b["diff_se"], 4),
                               "items_better": sum(e > o for o, e in f1), "items_worse": sum(e < o for o, e in f1)}
        if req.model != "original":
            summary["edited_tensors"] = sorted(edits)
        summary["pred_file"] = str(self.box.work / pred_name)
        if meta is not None:
            summary["arrays_dir"] = str(self.box.work / arrays_name)
            summary["arrays"] = {n: s for n, s in shapes.items() if "/" not in n}
            if req.attention:
                summary["attention_dirs"] = sorted({n.split("/")[0] for n in shapes if "/" in n})
        summary["seconds"] = round(time.monotonic() - start, 2)
        return Result(summary, pred, meta)

    def _publish(self, stage: Path, names: list[str]) -> None:
        """Move each staged output into work/, first moving out whatever the editor's code put at that name.

        Renames act on directory entries and never follow a symlink, and the server never writes into work/
        any other way, so a symlink planted there cannot redirect its writes."""
        try:
            work = open_dir(self.box.root, "work")
        except OSError:
            raise RequestError("work/ is not a real folder (a symlink or a file), so the results "
                               "cannot be written") from None
        staged = open_dir(stage)
        try:
            for name in names:
                try:
                    os.rename(name, f"replaced_{name}", src_dir_fd=work, dst_dir_fd=staged)
                except FileNotFoundError:
                    pass
                os.rename(name, name, src_dir_fd=staged, dst_dir_fd=work)
        finally:
            os.close(work)
            os.close(staged)

    def _capture(self, req: Request, texts: list[str], ids: list, out: Path, prefix: str, shapes: dict) -> dict:
        """Write the requested arrays for every text, in request order, from one unpadded pass per text."""
        out.mkdir(parents=True, exist_ok=True)
        encs = [self.tok(t, truncation=True, max_length=MAX_LEN, return_tensors="pt") for t in texts]
        lengths = [e["input_ids"].shape[1] for e in encs]
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
        n_tokens = int(sum(lengths))
        cfg = self.model.config
        token_arrays = {}
        if req.label_probs:
            token_arrays["label_probs"] = cfg.num_labels
        token_arrays |= {f"hidden_L{l}": cfg.hidden_size for l in req.hidden_states}
        token_arrays |= {f"mlp_L{l}": cfg.intermediate_size for l in req.mlp_act}
        mm = {}
        for name, d in token_arrays.items():
            mm[name] = open_memmap(out / f"{prefix}{name}.npy", mode="w+", dtype=np.float32, shape=(n_tokens, d))
            shapes[f"{prefix}{name}.npy"] = [n_tokens, d]
        for l in req.attention:
            (out / f"{prefix}attention_L{l}").mkdir(exist_ok=True)
        mlp_out: dict[int, torch.Tensor] = {}
        hooks = [self.model.bert.encoder.layer[l].intermediate.register_forward_hook(
            lambda mod, inp, o, l=l: mlp_out.__setitem__(l, o[0])) for l in req.mlp_act]
        attn_impl = self.model.config._attn_implementation
        if req.attention:
            self.model.set_attn_implementation("eager")
        try:
            for i, enc in enumerate(encs):
                o = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               token_type_ids=enc.get("token_type_ids"),
                               output_hidden_states=bool(req.hidden_states), output_attentions=bool(req.attention))
                rows = slice(starts[i], starts[i] + lengths[i])
                if req.label_probs:
                    mm["label_probs"][rows] = o.logits[0].softmax(-1).numpy()
                for l in req.hidden_states:
                    mm[f"hidden_L{l}"][rows] = o.hidden_states[l][0].numpy()
                for l in req.mlp_act:
                    mm[f"mlp_L{l}"][rows] = mlp_out[l].numpy()
                for l in req.attention:
                    a = o.attentions[l][0].numpy().astype(np.float32)
                    np.save(out / f"{prefix}attention_L{l}" / f"{i}.npy", a)
                    shapes[f"{prefix}attention_L{l}/{i}.npy"] = list(a.shape)
        finally:
            for h in hooks:
                h.remove()
            if req.attention:
                self.model.set_attn_implementation(attn_impl)
        for a in mm.values():
            a.flush()
        return {"texts": [{"index": i, "id": ids[i], "start_row": int(starts[i]), "length": lengths[i],
                           "tokens": self.tok.convert_ids_to_tokens(encs[i]["input_ids"][0].tolist()),
                           "word_ids": encs[i].word_ids(0)} for i in range(len(texts))]}

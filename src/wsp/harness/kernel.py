"""The editor's persistent Python kernel and the save_edit refusal rules.

Standalone: imports only the stdlib, numpy, torch, safetensors and tokenizers, because it is copied
into the sandbox and run there, where the project is unreadable. The forward-pass server imports
`refusal` from this same file, so both gates apply one set of rules.

Run as `python -I kernel.py <box> <request fd> <reply fd>`. Requests and replies are one JSON object
per line on the two pipe fds, so editor code that writes to fd 1 cannot corrupt the protocol. Each reply
echoes its request's nonce, so the host can tell a forged or stale reply line from the real one.
"""

import ast
import io
import json
import linecache
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

RTOL, ATOL = 1e-6, 1e-9
SCALE_SHARE = 0.99
OUTPUT_CAP = 200_000


def _unknown(name, new, orig):
    if orig is None:
        return f"{name!r} is not a tensor of this model; see NAMES"


def _embeddings(name, new, orig):
    if "embeddings" in name:
        return f"{name} is part of the embeddings, which are out of scope"


def _classifier(name, new, orig):
    if name.startswith("classifier."):
        return f"{name} is part of the classification head, which is out of scope"


def _shape(name, new, orig):
    if tuple(new.shape) != tuple(orig.shape):
        return f"shape {tuple(new.shape)} does not match {name}'s shape {tuple(orig.shape)}"


def _finite(name, new, orig):
    bad = int((~torch.isfinite(new)).sum())
    if bad:
        return f"{bad} values are not finite (nan or inf)"


def _constant_multiple(name, new, orig):
    """Refuse an edit that is exactly, or almost exactly, the whole tensor times one number.

    Rescaling the whole tensor and nudging a few entries would pass an exact test, so the rule asks how much of
    the change the best single scale factor accounts for. On this model's tensors that share is at most 58% for
    targeted edits (zeroing a head, scaling 64 columns, removing a direction, changing one entry), and above 99.7%
    for disguised rescales.
    """
    o, n = orig.double(), new.double()
    if torch.allclose(n, o, rtol=RTOL, atol=ATOL):
        return f"{name} is unchanged from the original"
    den = float((o * o).sum())
    c = float((n * o).sum()) / den if den else 0.0
    change = float(((n - o) ** 2).sum())
    explained = 1 - float(((n - c * o) ** 2).sum()) / change
    if explained >= SCALE_SHARE:
        return (f"the edit only or almost only multiplies the whole of {name} by a constant: scaling it by "
                f"c={c:.6g} accounts for {explained:.2%} of the change, and an edit that one scale factor accounts "
                f"for {SCALE_SHARE:.0%} or more of is out of scope")


RULES = (_unknown, _embeddings, _classifier, _shape, _finite, _constant_multiple)


def refusal(name: str, new: torch.Tensor, orig: torch.Tensor | None) -> str | None:
    """Why `new` may not replace tensor `name`, or None if it may. `orig` is None for an unknown name."""
    for rule in RULES:
        reason = rule(name, new, orig)
        if reason:
            return reason
    return None


class _Capped(io.StringIO):
    """Keeps the first OUTPUT_CAP characters so a runaway print loop cannot exhaust memory."""

    def __init__(self):
        super().__init__()
        self.total = 0

    def write(self, s):
        self.total += len(s)
        room = OUTPUT_CAP - self.tell()
        if room > 0:
            super().write(s[:room])
        return len(s)


class Session:
    def __init__(self, box: Path):
        from safetensors import safe_open
        from tokenizers import Tokenizer

        self.box = box
        self.weights = box / "model" / "model.safetensors"
        self.edited_dir = box / "work" / "edited"
        self.edited_dir.mkdir(parents=True, exist_ok=True)
        with safe_open(self.weights, "pt") as f:
            self.names = list(f.keys())
            self.originals = {n: f.get_tensor(n).float().contiguous() for n in self.names}
        tok = Tokenizer.from_file(str(box / "model" / "tokenizer.json"))
        # tokenizer.json pads every encoding to 512; the forward server pads nothing, so neither does tok.
        tok.no_padding()
        self.cell = 0
        self.ns = {"__name__": "__main__", "NAMES": list(self.names),
                   "load": self.load, "load_original": self.load_original, "save_edit": self.save_edit,
                   "drop_edit": self.drop_edit, "edited": self.edited,
                   "tok": tok}

    def _path(self, name: str) -> Path:
        return self.edited_dir / f"{name}.npy"

    def load_original(self, name: str) -> torch.Tensor:
        if name not in self.originals:
            raise KeyError(f"{name!r} is not a tensor of this model; see NAMES")
        return self.originals[name].clone()

    def load(self, name: str) -> torch.Tensor:
        path = self._path(name)
        if name in self.originals and path.exists():
            return torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
        return self.load_original(name)

    def save_edit(self, name: str, tensor) -> None:
        new = torch.as_tensor(tensor).detach().to("cpu", torch.float32).contiguous()
        reason = refusal(name, new, self.originals.get(name))
        if reason:
            raise ValueError(f"save_edit refused: {reason}")
        np.save(self._path(name), new.numpy())

    def drop_edit(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def edited(self) -> list[str]:
        return sorted(p.name[:-4] for p in self.edited_dir.glob("*.npy"))

    def run(self, code: str) -> tuple[str, int]:
        self.cell += 1
        filename = f"<cell {self.cell}>"
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        buf = _Capped()
        old = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf
        try:
            tree = ast.parse(code, filename)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, filename, "exec"), self.ns)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), filename, "eval"), self.ns)
                if value is not None:
                    print(repr(value))
        except BaseException as e:
            tb = e.__traceback__
            while tb is not None and not tb.tb_frame.f_code.co_filename.startswith("<cell"):
                tb = tb.tb_next
            traceback.print_exception(type(e), e, tb)
        finally:
            sys.stdout, sys.stderr = old
        return buf.getvalue(), buf.total


def main() -> None:
    box = Path(sys.argv[1])
    del sys.argv[1:]
    # Move the protocol off fds 0 and 1, so code that prints at the C level or reads stdin cannot touch it.
    fd_in, fd_out = os.dup(0), os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)
    os.close(devnull)
    torch.set_grad_enabled(False)
    os.chdir(box / "work")
    session = Session(box)
    reader = os.fdopen(fd_in, "r", encoding="utf-8")
    writer = os.fdopen(fd_out, "w", encoding="utf-8")
    writer.write(json.dumps({"ready": True}) + "\n")
    writer.flush()
    for line in reader:
        start = time.monotonic()
        request = json.loads(line)
        output, total = session.run(request["code"])
        writer.write(json.dumps({"nonce": request["nonce"], "output": output, "total": total,
                                 "seconds": time.monotonic() - start}) + "\n")
        writer.flush()


if __name__ == "__main__":
    main()

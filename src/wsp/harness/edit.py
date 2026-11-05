"""The submitted edit as a sparse diff against the original weights, and its exact rebuild."""

import errno
import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from wsp.harness.box import Box, open_dir
from wsp.harness.kernel import refusal


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


class Unreadable(Exception):
    """An edit file the server will not load. The message never quotes the file's bytes."""


def _load_npy(dir_fd: int, entry: str) -> np.ndarray:
    try:
        fd = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except OSError as e:
        raise Unreadable("is a symbolic link" if e.errno == errno.ELOOP else "cannot be opened") from None
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise Unreadable("is not a regular file")
        if st.st_nlink != 1:
            raise Unreadable("is a hard link")
        try:
            a = np.load(f, allow_pickle=False)
            if isinstance(a, np.ndarray) and a.dtype.kind in "fiu":
                return a.astype(np.float32)
        except Exception:
            pass
    raise Unreadable("is not a .npy file holding one numeric array")


def read_edits(box: Box, originals: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Every work/edited/*.npy as float32, and the refusal reason of each file that fails the save_edit rules.

    Raises ValueError if work/ or work/edited is a symlink or a file, since nothing under work/ is trusted."""
    try:
        dir_fd = open_dir(box.root, "work", "edited")
    except FileNotFoundError:
        return {}, {}
    except OSError:
        raise ValueError("work/ or work/edited is not a real folder (a symlink or a file)") from None
    edits, refused = {}, {}
    try:
        for entry in sorted(e for e in os.listdir(dir_fd) if e.endswith(".npy")):
            name = entry[:-4]
            try:
                new = _load_npy(dir_fd, entry)
            except Unreadable as e:
                refused[name] = f"{entry} {e}"
                continue
            orig = originals.get(name)
            reason = refusal(name, torch.from_numpy(new), None if orig is None else torch.from_numpy(orig))
            if reason:
                refused[name] = reason
            else:
                edits[name] = new
    finally:
        os.close(dir_fd)
    return edits, refused


def write_edit(box: Box, orig_path: Path, out_dir: Path) -> dict:
    """Write out_dir/edit.npz and out_dir/manifest.json. Raises ValueError if any edited file breaks the rules."""
    originals = load_file(orig_path)
    edits, refused = read_edits(box, originals)
    if refused:
        raise ValueError(f"refused edits: {refused}")
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, tensors = {}, []
    for name, new in edits.items():
        orig = originals[name].astype(np.float32)
        idx = np.flatnonzero(new.view(np.uint32) != orig.view(np.uint32)).astype(np.int64)
        arrays[f"{name}::idx"] = idx
        arrays[f"{name}::val"] = new.reshape(-1)[idx]
        tensors.append({"name": name, "shape": list(new.shape), "entries_changed": int(idx.size),
                        "entries": int(new.size), "sha256_original": sha256(orig), "sha256_edited": sha256(new)})
    np.savez(out_dir / "edit.npz", **arrays)
    manifest = {"tensors": tensors, "n_tensors": len(tensors),
                "entries_changed": sum(t["entries_changed"] for t in tensors)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def apply_npz(npz_path: Path, orig_path: Path) -> dict[str, np.ndarray]:
    """The full edited tensors, rebuilt from the diff and the original weights."""
    originals = load_file(orig_path)
    out = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for key in z.files:
            name, _, part = key.rpartition("::")
            if part != "idx":
                continue
            new = originals[name].astype(np.float32).copy()
            new.reshape(-1)[z[key]] = z[f"{name}::val"]
            out[name] = new
    return out

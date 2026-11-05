"""The editor's sandbox folders and the builder's run folder."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from wsp.data import MODEL_DIR, ROOT, SETS

SANDBOX_ROOT = Path.home() / "wsp-sandbox"
RUNS_ROOT = ROOT / "runs"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt")
PRACTICE_FILE = "sg_train.jsonl"
KERNEL_SRC = Path(__file__).with_name("kernel.py")


@dataclass(frozen=True)
class Box:
    """Everything the kernel may see, under one folder outside the project. Paths are realpaths."""
    root: Path

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def weights(self) -> Path:
        return self.model / "model.safetensors"

    @property
    def practice(self) -> Path:
        return self.root / "practice"

    @property
    def practice_file(self) -> Path:
        return self.practice / PRACTICE_FILE

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def edited(self) -> Path:
        return self.work / "edited"

    @property
    def kernel(self) -> Path:
        return self.root / "kernel"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def cli(self) -> Path:
        return self.root / "cli"

    @property
    def staging(self) -> Path:
        """Where the server builds its outputs before moving them into work/. The kernel cannot reach it."""
        return self.root / "staging"


@dataclass(frozen=True)
class RunDir:
    """The builder's log folder for one run, inside the project."""
    root: Path

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.jsonl"

    @property
    def forward(self) -> Path:
        return self.root / "forward"

    @property
    def edit(self) -> Path:
        return self.root / "edit"

    @property
    def final_answer(self) -> Path:
        return self.root / "final_answer.md"

    @property
    def context_check(self) -> Path:
        return self.root / "context_check.md"

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def kernel_log(self) -> Path:
        return self.root / "kernel.log"


def open_dir(root: Path, *names: str) -> int:
    """An fd for root/names[0]/names[1]/..., opened one level at a time so no symlink below `root` is followed.

    The kernel can replace anything under work/, work itself included, with a symlink. Raises OSError
    (ELOOP for a symlink, ENOTDIR for a file, ENOENT for nothing there)."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _rmtree(path: Path) -> None:
    """Delete path without following a symlink at it or in it."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _copy(files: list[Path], dest: Path) -> None:
    _rmtree(dest)
    dest.mkdir(parents=True)
    for f in files:
        shutil.copyfile(f, dest / f.name)


def setup_box(name: str, fresh: bool = False, root: Path = SANDBOX_ROOT) -> Box:
    """Build or refresh the box. The model, practice and kernel copies are always remade; work/ survives unless
    `fresh`. srt's write allowlist, not file modes, keeps the kernel from changing the copies."""
    box = Box((root / name).resolve())
    box.root.mkdir(parents=True, exist_ok=True)
    _copy([MODEL_DIR / f for f in MODEL_FILES], box.model)
    _copy([SETS / PRACTICE_FILE], box.practice)
    _copy([KERNEL_SRC], box.kernel)
    if fresh or box.work.is_symlink() or not box.work.is_dir():
        _rmtree(box.work)
    _rmtree(box.tmp)
    _rmtree(box.staging)
    for d in (box.edited, box.tmp, box.cli, box.staging):
        d.mkdir(parents=True, exist_ok=True)
    return box

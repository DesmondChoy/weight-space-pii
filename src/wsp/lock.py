"""The M5 freeze: hashes of everything that decides a score, written once to lock.json and checked before use."""

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

from wsp.data import ROOT
from wsp.harness.driver import MAX_PYTHON_RUNS, Editor, cli_version, run

LOCK = ROOT / "lock.json"
RENDERED_PROMPT = "docs/m5_task_prompt.txt"
GROUPS: dict[str, tuple[str, ...]] = {
    "sets": ("data/sets/sg_train.jsonl", "data/sets/sg_eval.jsonl", "data/sets/nemotron_eval.jsonl",
             "data/sets.json"),
    "scoring": ("src/wsp/score.py", "src/wsp/predict.py", "src/wsp/data.py"),
    "forward_pass_server": ("src/wsp/harness/server.py", "src/wsp/harness/edit.py", "src/wsp/harness/kernel.py",
                            "src/wsp/harness/box.py"),
    "prompt": ("src/wsp/harness/task_prompt.txt", "src/wsp/harness/prompt.py", RENDERED_PROMPT),
    "model": ("data/model/config.json", "data/model/model.safetensors", "data/model/tokenizer.json",
              "data/model/tokenizer_config.json", "data/model/special_tokens_map.json", "data/model/vocab.txt"),
    "environment": ("uv.lock", "src/wsp/harness/kernel_host.py"),
    "baseline": ("scripts/baseline.py", "src/wsp/outcomes.py", "docs/m5_prediction.md"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def node_version() -> str:
    return subprocess.run(["node", "--version"], capture_output=True, text=True, check=True).stdout.strip()


def run_settings(name: str, budget: int, effort: str = "max", max_turns: int | None = None) -> dict:
    """Everything the driver decides about what Opus 5.5 sees, other than the prompt: the exact CLI command
    without its session ID, the CLI's environment and version, and the limits the prompt states."""
    info = Editor(Path(), print, effort=effort).info()
    cmd = info["cli_command"]
    return {"name": name, "budget": budget, "max_turns": max_turns, "cli_command": cmd[:cmd.index("--session-id")],
            "cli_env": info["cli_env"], "cli_env_passed": info["cli_env_passed"], "cli_version": info["cli_version"], "max_python_runs": MAX_PYTHON_RUNS,
            "kernel_timeout": inspect.signature(run).parameters["kernel_timeout"].default}


def snapshot(root: Path = ROOT, groups: dict[str, tuple[str, ...]] = GROUPS) -> dict[str, dict[str, str]]:
    return {g: {p: sha256_file(root / p) for p in paths} for g, paths in groups.items()}


def write(settings: dict, root: Path = ROOT, groups: dict[str, tuple[str, ...]] = GROUPS) -> dict:
    """Write lock.json with the run's settings. Refuses to replace an existing lock: the freeze happens once."""
    path = root / LOCK.name
    if path.exists():
        raise FileExistsError(f"{path} exists; the freeze is final")
    lock = {"run": settings, "node": node_version(), "files": snapshot(root, groups)}
    path.write_text(json.dumps(lock, indent=1) + "\n")
    return lock


def read(root: Path = ROOT) -> dict:
    return json.loads((root / LOCK.name).read_text())


def drift(root: Path = ROOT) -> list[str]:
    """Every way the tree differs from lock.json, as one line each. Empty when nothing changed."""
    lock = read(root)
    out = []
    for group, files in lock["files"].items():
        for p, want in files.items():
            if not (root / p).is_file():
                out.append(f"{group}: {p} is missing")
            elif sha256_file(root / p) != want:
                out.append(f"{group}: {p} changed")
    if (node := node_version()) != lock["node"]:
        out.append(f"environment: node is {node}, locked {lock['node']}")
    return out


def check(root: Path = ROOT) -> dict:
    """Return lock.json's content, or raise if anything it froze has changed."""
    if problems := drift(root):
        raise RuntimeError("the tree differs from lock.json:\n" + "\n".join(problems))
    return read(root)

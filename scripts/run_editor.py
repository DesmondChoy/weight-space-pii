"""Run one editor session through the M4 harness. This calls the real editor CLI and costs money.

The run that lock.json names refuses to start unless the tree matches the lock, its settings (CLI command,
CLI version, limits) match the locked ones, and the prompt it would send is byte for byte the frozen one.
"""

import argparse
import json

from wsp.data import ROOT
from wsp.harness.box import RUNS_ROOT, setup_box
from wsp.harness.driver import _git_head, run
from wsp.harness.prompt import render
from wsp.lock import LOCK, RENDERED_PROMPT, check, run_settings


def check_locked_run(name: str, budget: int, effort: str, max_turns: int | None, overwrite: bool) -> None:
    if not LOCK.exists() or name != (lock := check())["run"]["name"]:
        return
    if _git_head()["dirty"]:
        raise RuntimeError(f"the git tree has uncommitted changes; {name} must start from a commit")
    if (RUNS_ROOT / name).exists() and not overwrite:
        raise FileExistsError(f"{RUNS_ROOT / name} exists; pass --overwrite to replace it")
    settings = run_settings(name, budget, effort, max_turns)
    if differ := sorted(k for k in settings.keys() | lock["run"].keys() if settings.get(k) != lock["run"].get(k)):
        raise RuntimeError(f"{name} differs from its locked settings in: {', '.join(differ)}")
    if render(budget, setup_box(name, fresh=True)) != (ROOT / RENDERED_PROMPT).read_text():
        raise RuntimeError(f"the prompt for {name} differs from {RENDERED_PROMPT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="run name: runs/<name>/ and ~/wsp-sandbox/<name>/")
    ap.add_argument("--budget", type=int, required=True, help="forward passes")
    ap.add_argument("--max-turns", type=int, help="editor replies, counting the final one (default: no limit)")
    ap.add_argument("--effort", default="max")
    ap.add_argument("--ask-context", action="store_true", help="ask the editor what tools and context it has")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing runs/<name>/")
    a = ap.parse_args()
    check_locked_run(a.name, a.budget, a.effort, a.max_turns, a.overwrite)
    result = run(a.name, a.budget, a.max_turns, a.effort, a.ask_context, overwrite=a.overwrite)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

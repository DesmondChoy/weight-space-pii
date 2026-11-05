"""M5 freeze: render the 50-pass task prompt for the real run, then write lock.json. Runs once.

The locked run uses the driver's defaults: effort max and no turn cap.
"""

import argparse

from wsp.data import ROOT
from wsp.harness.box import _rmtree, setup_box
from wsp.harness.prompt import render
from wsp.lock import GROUPS, RENDERED_PROMPT, run_settings, write


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="p50", help="the M6 run name, which sets the box paths in the prompt")
    ap.add_argument("--budget", type=int, default=50)
    a = ap.parse_args()
    box = setup_box(a.run, fresh=True)
    try:
        (ROOT / RENDERED_PROMPT).write_text(render(a.budget, box))
    finally:
        _rmtree(box.root)
    lock = write(run_settings(a.run, a.budget))
    n = sum(len(f) for f in lock["files"].values())
    print(f"wrote {RENDERED_PROMPT} and lock.json: {n} files in {len(GROUPS)} groups, node {lock['node']}")


if __name__ == "__main__":
    main()

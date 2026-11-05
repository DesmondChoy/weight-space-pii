"""Run one editor session through the M4 harness. This calls the real editor CLI and costs money."""

import argparse
import json

from wsp.harness.driver import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="run name: runs/<name>/ and ~/wsp-sandbox/<name>/")
    ap.add_argument("--budget", type=int, required=True, help="forward passes")
    ap.add_argument("--max-turns", type=int, help="editor replies, counting the final one (default: no limit)")
    ap.add_argument("--effort", default="max")
    ap.add_argument("--ask-context", action="store_true", help="ask the editor what tools and context it has")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing runs/<name>/")
    a = ap.parse_args()
    result = run(a.name, a.budget, a.max_turns, a.effort, a.ask_context, overwrite=a.overwrite)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

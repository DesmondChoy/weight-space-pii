# AGENTS.md

This repo runs a small version of MIT CSAIL's "Sparks of Weight-Space Programming", Experiment 8: an LLM editor edits the weights of a MiniLM PII detector to improve it on Singapore text. The brief in [docs/weight-space-pii-handoff.md](docs/weight-space-pii-handoff.md) sets the initial scope.

## Decisions override the brief

The milestone reports, `docs/m<N>_report.md`, record the latest decisions, and they may deviate from the brief. Where a report and the brief disagree, follow the report. Where two reports disagree, follow the later one. For example, the M2 report narrows the scored labels and changes the main eval size that the brief's §3 sets.

## Git

- Work on `main`. Don't create branches or worktrees unless the user asks for one.
- Commit or push only when the user asks.

## Picking up work

The work follows eight milestones, M1 to M8, in §7 of the brief. To find the next step:

1. Read the brief, §7 in particular.
2. Read every milestone report for the decisions that change the brief.
3. Find the latest milestone report, `docs/m<N>_report.md`. Its status line says whether that milestone is complete, and its "Open items" section lists what is left.
4. Check `git log` and `git status` for work done since that report.
5. The next milestone is M<N+1>, unless the latest report still has open items.

The brief requires a stop at the end of every milestone. Show the user what the milestone lists under "Show me", and don't start the next milestone until the user says to. Don't change anything an earlier milestone froze without asking.

Each milestone ends with its own report at `docs/m<N>_report.md`, in the style of [docs/m1_report.md](docs/m1_report.md).

## Environment

- Python is managed with `uv`, and every dependency is pinned in `pyproject.toml` and `uv.lock`. Run scripts with `uv run python -I scripts/<name>.py`.
- `data/` is gitignored except `data/revisions.json`. `scripts/fetch.py` rebuilds it at the pinned revisions.

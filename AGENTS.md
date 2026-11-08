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
2. Read every milestone report for the decisions that change the brief. Also read [results/p50b_report.md](results/p50b_report.md), which records decisions made after M7. `results/` holds the reports, logbooks and figures of the runs `p50` and `p50b`.
3. Find the latest milestone report, `docs/m<N>_report.md`. Its status line says whether that milestone is complete, and its "Open items" section lists what is left.
4. Check `git log` and `git status` for work done since that report.
5. The next milestone is M<N+1>, unless the latest report still has open items.

The brief requires a stop at the end of every milestone. Show the user what the milestone lists under "Show me", and don't start the next milestone until the user says to. Don't change anything an earlier milestone froze without asking.

Each milestone ends with its own report at `docs/m<N>_report.md`, in the style of [docs/m1_report.md](docs/m1_report.md).

## Opening a milestone report

A reader opens a report to understand one milestone without rereading the brief and every earlier report. So each report opens, right after its title, one-paragraph summary and status line, with a refresher section and at least one figure. [docs/m4_report.md](docs/m4_report.md) shows the pattern.

- **Write a refresher section.** Title it for what the reader needs, such as "What you need to know first". Write it as a numbered list, where each step builds on the one before. Restate only what this milestone depends on: the experiment's question, decisions from earlier reports, and the terms the rest of the report uses. End with the milestone's own job. Bold each step's first sentence.
- **Name the actors plainly.** Say "Opus 5.5" for the model that edits and "MiniLM" for the model being edited. Explain any role name from the brief, such as "the editor", the first time it appears.
- **Add at least one figure.** Draw what the milestone builds or decides, such as how the parts connect, where a rule applies, or how data flows. Generate each figure as an SVG from a script in `docs/figures/`. Add new figures to [docs/figures/report_figures.py](docs/figures/report_figures.py), which holds the shared style. Never edit an SVG by hand.
- **Explain each figure.** Give it alt text that states what it shows. Follow it with a short paragraph that says how to read it. Name the script that generates it.
- **Check each figure.** Open the rendered SVG at full width and make sure no text overlaps or is cut off. Rerun the script and confirm the existing figures don't change.

## Environment

- Python is managed with `uv`, and every dependency is pinned in `pyproject.toml` and `uv.lock`. Run scripts with `uv run python -I scripts/<name>.py`.
- `data/` is gitignored except `data/revisions.json`. `scripts/fetch.py` rebuilds it at the pinned revisions.

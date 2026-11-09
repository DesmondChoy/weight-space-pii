# Weight-space programming on a PII detector

A small version of Experiment 8 from MIT CSAIL's [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Opus 5.5 reads the weights of MiniLM, a small PII detector trained on American data, and edits them directly to make MiniLM better on Singapore text. No gradient descent or optimizer runs.

## Result

Opus 5.5 found two repairs, and a second run with the same settings found the same two.

| | Unedited | `p50` | `p50b` |
|---|---|---|---|
| Singapore F1 (5,004 hidden rows) | 0.865 | 0.924 | 0.923 |
| NRICs without a cue word, share right | 0% | 85% | 93% |
| Targeted titles tagged as a name | 909 | 2 | 14 |
| Nemotron F1, all 55 labels (18,515 rows) | 0.952 | 0.952 | 0.952 |
| Nemotron `national_id` F1 | 0.960 | 0.913 | 0.889 |

- **Title repair.** MiniLM stops tagging titles such as Madame and Mstr as names. This gives about three quarters of the gain.
- **NRIC repair.** MiniLM tags an NRIC as `national_id` from its shape, without a nearby cue word.
- **Cost.** On Nemotron, biometric IDs shaped like a letter and 11 digits are now tagged `national_id`.

Whether this matches the paper's finding, that edits can remove an obstacle but can't add a missing ability, is still an open question. [docs/m7_report.md](docs/m7_report.md) and [results/p50b_report.md](results/p50b_report.md) give the details.

## Where to read

- [docs/weight-space-pii-handoff.md](docs/weight-space-pii-handoff.md): the brief, with the setup and the eight milestones.
- `docs/m1_report.md` to `docs/m7_report.md`: one report per milestone. Each later report overrides the brief and the earlier reports where they disagree.
- [results/](results/): the `p50b` report, the logbooks of both runs (`p50_logbook.html`, `p50b_logbook.html`) and the M7 review page.
- [AGENTS.md](AGENTS.md): how to pick up work in this repo.

## Layout

| Path | Contents |
|---|---|
| `src/wsp/` | Data loading, prediction, scoring, and the lock check |
| `src/wsp/harness/` | The kernel, forward-pass server, edit rules and driver that run an Opus 5.5 session |
| `scripts/` | One script per step: fetch, build sets, baseline, run, evaluate, review |
| `lock.json` | Hashes of every file that decides a score, frozen at M5 |
| `docs/figures/` | Scripts that generate every report figure |
| `tests/` | Unit tests for scoring, the lock and the harness |

`data/` and `runs/` are gitignored. Only `data/revisions.json` and `data/sets.json` are committed.

## Setup

Python 3.13 and every dependency are pinned in `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

Download the model and datasets at the pinned revisions:

```bash
uv run python -I scripts/fetch.py
```

Build the practice, main eval and regression sets:

```bash
uv run python -I scripts/build_sets.py
```

Run the tests:

```bash
uv run python -I -m unittest discover -s tests
```

## Running

A run calls the Claude Code CLI as Opus 5.5 and costs money. It refuses to start unless the tree matches `lock.json`.

```bash
uv run python -I scripts/run_editor.py --name p50 --budget 50
```

Score a run on the hidden sets:

```bash
uv run python -I scripts/evaluate.py --run p50
```

A repeat run uses `--replicate-of p50` with both scripts. [docs/m6_report.md](docs/m6_report.md) and [docs/m7_report.md](docs/m7_report.md) list the exact commands each milestone ran.

# Weight-space programming on a PII detector

A small version of Experiment 8 from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), a blog post by four researchers at MIT CSAIL. Opus 5.5 reads the weights of MiniLM, a small PII detector trained on American data, and edits them directly to make MiniLM better on Singapore text. No gradient descent or optimizer runs. Opus 5.5 does fit small detectors to 1,000 practice rows with answers, each in one closed-form calculation, and it chooses between candidate edits by their F1 on those rows.

The Singapore text is the Singapore rows of openpii, a synthetic dataset. The hidden rows that grade each edit come from the same dataset as the practice rows, so the result describes this dataset, not Singapore text in general.

## Result

Opus 5.5 found two repairs, and a second run with the same settings found the same two.

| | Unedited | `p50` | `p50b` |
|---|---|---|---|
| Singapore F1 (5,004 hidden rows) | 0.865 | 0.924 | 0.923 |
| The same score, if tagging a title as a name isn't a mistake | 0.901 | 0.927 | 0.926 |
| NRICs without a cue word, share right | 0% | 85% | 93% |
| Targeted titles tagged as a name | 909 | 2 | 14 |
| Nemotron F1, all 55 labels (18,515 rows) | 0.952 | 0.952 | 0.952 |
| Nemotron `national_id` F1 | 0.960 | 0.913 | 0.889 |

- **Title repair.** MiniLM stops tagging titles such as Madame and Mstr as names. This gives about three quarters of the gain.
- **NRIC repair.** MiniLM tags an NRIC as `national_id` from its shape, without a nearby cue word. The task prompt states an NRIC's format as part of the scoring rules, so Opus 5.5 didn't have to discover it. The prompt says nothing about titles.
- **Cost.** On Nemotron, biometric IDs shaped like a letter and 11 digits are now tagged `national_id`.
- **How much the title repair counts depends on the scoring.** The scoring, frozen before the runs, counts "Madame" tagged as a name as a mistake, because a title is not a name. But openpii marks titles as personal information of another type, and masking a title leaks nothing. If that tag isn't counted as a mistake, the gain falls from +0.059 to +0.026. The second row of the table gives both runs under that rule.

In the paper's Experiment 8, editing could remove an obstacle that hid an ability, but it could not add an ability the model lacked. Whether this result fits that pattern is still an open question. [docs/m7_report.md](docs/m7_report.md) and [results/p50b_report.md](results/p50b_report.md) give the details.

## Where to read

- [docs/weight-space-pii-handoff.md](docs/weight-space-pii-handoff.md): the brief, with the setup and the eight milestones.
- `docs/m1_report.md` to `docs/m7_report.md`: one report per milestone. Each later report overrides the brief and the earlier reports where they disagree.
- [results/](results/): the `p50b` report, each run's final answer and edit manifest, the logbooks of both runs (`p50_logbook.html`, `p50b_logbook.html`) and the M7 review page.
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

You need macOS, [uv](https://docs.astral.sh/uv/) and Node.js. The harness runs Opus 5.5's code in a sandbox, Anthropic's `sandbox-runtime`, which it starts through `npx` with a macOS Seatbelt profile. The harness tests start that sandbox too. Python 3.13 and every dependency are pinned in `pyproject.toml` and `uv.lock`.

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

The locked runs `p50` and `p50b` replay only on the machine that made them. `lock.json` pins Node.js v22.23.4 and Claude Code 2.1.318, and the frozen prompt holds that machine's sandbox path. `runs/` isn't published either, so `scripts/evaluate.py` can't rescore them from a clone. Every run first checks the tree against `lock.json`, so a run on another machine needs that Node.js version. With it, a run under a new name and without `--replicate-of` starts, because only the locked run and its repeats get the settings and prompt checks. The commands below are the ones this repo ran.

```bash
uv run python -I scripts/run_editor.py --name p50 --budget 50
```

Score a run on the hidden sets:

```bash
uv run python -I scripts/evaluate.py --run p50
```

A repeat run uses `--replicate-of p50` with both scripts. [docs/m6_report.md](docs/m6_report.md) and [docs/m7_report.md](docs/m7_report.md) list the exact commands each milestone ran.

## Data, licences and credits

Both datasets are synthetic. The names, ID numbers and addresses quoted in the reports and logbooks belong to no real person.

| Material | Source | Licence | Use in this repo |
|---|---|---|---|
| The experiment's design and the task prompt | [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, MIT CSAIL, and its logs at [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the logs and prompts | `src/wsp/harness/task_prompt.txt` is adapted from the Experiment 8 prompt, `exp8_math/runs/p50c/turns/t001/prompt.txt`. The model, task, data and scoring rules are changed. The logbooks and `docs/m4_dry_run_transcript.md` quote the adapted prompt |
| Singapore rows | [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) by Ai4Privacy / Ai Suisse SA | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Rows with `region == "SG"`, relabelled into MiniLM's label names. The reports, figures and logbooks quote rows, with annotations added |
| Regression rows | [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) by NVIDIA | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | MiniLM's held-out split, rebuilt from its training notebook. `docs/m7_changed_rows.md` and `results/m7_review.html` quote rows, with annotations added |
| MiniLM, its model card and its training notebook | [Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection](https://huggingface.co/Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection) by Negative Star Innovators | MIT, in [docs/sources/LICENSE-local-pii-detector-minilm](docs/sources/LICENSE-local-pii-detector-minilm) | Copies of the model card and the notebook are in `docs/sources/`. The weights are downloaded, not committed |

The code and reports written for this repo are under the [MIT licence](LICENSE). The quoted dataset rows and the adapted prompt stay under their own licences.

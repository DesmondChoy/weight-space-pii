# M5 report: freeze and baseline

M5 freezes everything that decides a score, then scores the unedited MiniLM on every set. It writes `lock.json`, checks that the Singapore data leaves something to repair, records a prediction for each category, and estimates what the 50-pass run in M6 will cost. It follows §4, §5 and §7 of [weight-space-pii-handoff.md](weight-space-pii-handoff.md), with the decisions in the M2, M3 and M4 reports.

Status: built and run on 2026-11-05, and committed on `main` in six commits. Re-frozen in a seventh commit after a second review. Waiting for Desmond to review the items under "Show me".

## What you need to know first

The brief and four earlier reports hold the background. This section restates what M5 depends on. Each step builds on the one before it.

1. **Two AI models take part.** MiniLM (`MiniLM-L6-finetuned-pii-detection`) is the small model being edited. It tags each word of a text as a type of personal information, such as `first_name` or `national_id`, or as nothing. Opus 5.5 does the editing. The brief calls Opus 5.5 "the editor". It reads MiniLM's weights, runs experiments through a harness, and changes numbers inside MiniLM.
2. **The question is whether direct weight edits help.** MiniLM learned from American data, Nemotron-PII, and makes mistakes on Singapore text. The paper this experiment copies found that weight edits can remove an obstacle that hides an ability a model already has, but cannot add a missing one. This experiment asks whether the same holds for MiniLM.
3. **MiniLM was trained before this experiment, and nothing is trained during it.** MiniLM's makers fine-tuned it on 180,000 rows of Nemotron-PII and held back the other 20,000. In this experiment Opus 5.5 changes MiniLM's weights by hand. No step runs a training algorithm.
4. **Opus 5.5 works on a practice set.** The practice set is 1,000 rows of Singapore text, each with its answers. Opus 5.5 runs MiniLM on these rows and uses the answers to check whether its edits help. Nothing trains on them, so in machine-learning terms the practice set is a development set, not training data.
5. **The practice set is a small sample from openpii.** The Singapore text comes from openpii, a dataset MiniLM never saw. Its `train` split has 20,211 usable rows. M3 drew 1,000 of them at random with a fixed seed, a number of rows Opus 5.5 can work through. The other 19,211 rows are not used.
6. **Two hidden test sets measure the result.** Opus 5.5 never sees either one. The main eval asks whether the edit helped on Singapore text. It is every usable row of openpii's `validation` split, 5,004 rows. The regression check asks whether the edit broke anything on text like MiniLM's training data. It is the 18,515 of Nemotron's 20,000 held-out rows that fit in MiniLM's 512-token limit. Each set uses every usable row, so its standard error is small.

![Nemotron-PII's 180,000 training rows trained MiniLM and 18,515 held-out rows form the regression check, while openpii's Singapore train split gives a 1,000-row practice set and its validation split gives the 5,004-row main eval. A dotted line surrounds the three sets, which M5 freezes and scores](figures/set_roles.svg)

The figure shows points 3 to 6. Read each band from left to right, from source dataset, through the filter, to the set. The top band is Nemotron-PII, MiniLM's own data. Its purple training split shaped MiniLM before this experiment (point 3), and its held-out split becomes the regression check (point 6). The lower two bands are openpii's Singapore rows. The train split gives the green practice set (points 4 and 5), the only set Opus 5.5 can open, and the validation split gives the main eval (point 6). The two red sets are hidden from Opus 5.5.

The dotted line marks M5's part. Everything outside it existed before M5: MiniLM's training, and the filters and samples from M2 and M3. M5 hashes the three sets inside it into `lock.json` and scores the unedited MiniLM on each. The regression check's 0.952 covers all of MiniLM's labels, because an edit could break a label, such as `email`, that the Singapore sets do not score. The gate further down compares the same three labels on both datasets, so it shows Nemotron at 0.984. [docs/figures/report_figures.py](figures/report_figures.py) generates this figure and the others in this report.

7. **Three labels count on Singapore text.** They are `first_name`, `last_name` and `national_id`, where `national_id` means an NRIC. M3 set two scoring rules. Names are scored one word at a time, and a `tax_id` tag on an NRIC counts as right.
8. **M3 found that the NRIC failure depends on the words before it.** When "tax" appears in the 60 characters before an NRIC, MiniLM tags it `tax_id`, which scores as right. Without "tax" there, MiniLM got none of the practice NRICs right. M3 named this the likely target of an edit.
9. **M4 built the harness and dry-ran it.** The harness is the driver, a sandboxed Python kernel, the forward-pass server, and the `save_edit` rules. Two dry runs with Opus 5.5 showed it works end to end, and recorded the tokens and dollars each turn costs.
10. **M5's job is to freeze the experiment and measure the starting point.** After M5, nothing that decides a score or what Opus 5.5 sees may change. M5 hashes the sets, the scoring code, the server, the prompt, MiniLM's files, the environment and the baseline code into `lock.json`, together with the settings the run will use. It then scores the unedited MiniLM. That baseline is what M7 compares the edited MiniLM against.

## Decisions

These choices fill gaps in the brief. Desmond can change any of them before M6.

| Item | Brief | Decision |
|---|---|---|
| Restart sentence in the prompt | §6 says `notes.md` "is shown to you again if the conversation is restarted", and §2 step 5 builds the restart only if a run fills the context | Keep the sentence, word for word as in the brief and the paper. The restart lives in the driver, which the lock does not cover, so it can still be built before or during M6 |
| What the lock covers | Eval sets, scoring code, forward-pass server, prompt | Those four, plus MiniLM's files, the environment (`uv.lock`, the `srt` pin in `kernel_host.py`, and the Node version), the code that defines the prediction's categories, the prediction itself, and the run settings. This closes the M4 open item that `srt` sat outside `uv.lock` |
| The driver | Not named | Not hashed, so the restart can still be built. The lock instead records every value the driver passes to Opus 5.5 or states in the prompt: the exact `claude` command, its environment, the CLI version, 300 code runs, 900 s per run, and no turn cap |
| Run name | Not set | `p50`, after the paper's `p50c`. The name sets the paths in the prompt, so the lock fixes it |
| Gate standard error | "twice its bootstrap standard error" | Check both readings: the Singapore F1's own standard error, and the standard error of the gap. The gate passes only if both pass |
| Where the prediction lives | The run folder | [docs/m5_prediction.md](m5_prediction.md). `runs/` is gitignored, and the driver refuses to start a run whose folder exists. M6 copies the file into `runs/p50/` |
| Failing rows to show | Ask Desmond | 12, Desmond's choice |
| Explaining each step | §6 step 5 opens "Before each code block, explain the step in plain English" | Desmond's choice. The paper's "EXPLAIN EVERY STEP" paragraph replaces that sentence, word for word and from turn 1. The paper added it mid-run, at turn 29 of `p50c`, and its final prompt holds it between HOW TO WORK and RULES, as ours does |

## The lock

[scripts/freeze.py](../scripts/freeze.py) renders the 50-pass prompt into [docs/m5_task_prompt.txt](m5_task_prompt.txt), then writes `lock.json` through [src/wsp/lock.py](../src/wsp/lock.py). It hashes 25 files in seven groups and records the run settings. `lock.json` has SHA-256 `6b5d6ba5…0704c8`, and `runs/base/baseline.json` records the same hash. `wsp.lock.write` refuses to replace an existing lock, so the freeze happens once.

The run settings are what `wsp.lock.run_settings` returns. It builds the driver's own `Editor` and takes the exact `claude` command it would run, without the session ID. It adds the environment, the output of `claude --version` (2.1.292), the cap of 300 code runs, the 900 s kernel timeout and the turn cap, which is none. `scripts/run_editor.py` recomputes these for the run it is asked to start. It refuses `p50` if any of them differ, if `runs/p50/` already exists without `--overwrite`, or if the rendered prompt differs from the frozen one. It checks the folder before it touches the box, so an accidental rerun cannot wipe Opus 5.5's `work/` folder. A check against the real lock refused effort `high`, budget 49 and a turn cap of 30, each with the name of the setting that differed, and allowed the locked settings.

![Seven groups of files and the run settings go into lock.json, which the M5 baseline, the M6 run and the M7 evaluation each check](figures/lock_checks.svg)

This figure shows what the lock covers and who checks it. The purple boxes at the top are seven groups of files and the run settings. All of them go into `lock.json`, one SHA-256 hash per file plus the settings themselves. The three boxes at the bottom are the steps that compare the working tree with `lock.json` before they run. The green ones exist now. `scripts/baseline.py` refuses to score if any file differs. `scripts/run_editor.py` refuses to start the locked run unless its settings match too, and it renders the prompt it would send and compares it with the frozen copy. M7's scoring script does not exist yet and must call the same check.

The rendered prompt differs from the M4 dry-run prompt in three ways. The budget is 50 instead of 2, and the box name is `p50` instead of `dry2`, the two changes the M4 report expected. The third is the paper's "EXPLAIN EVERY STEP" paragraph, added by the fourth freeze below.

`wsp.lock.check` compares every hash and the Node version, and raises with a list of what differs. [tests/test_lock.py](../tests/test_lock.py) covers an unchanged tree, a changed file, a missing file, and a second freeze.

### The lock was written four times

The first freeze hashed 22 files in six groups. An independent review found that it left out the run settings, so `--effort high` would have passed every check. It also left out the code that defines the prediction's categories. Nothing was committed, so M5 deleted the first lock and froze again with the settings and the `baseline` group added. All 22 hashes from the first lock are unchanged in the second. The baseline was rerun against the second lock and gave identical scores, gate and per-kind tables.

The second lock was committed. A later review, described under "Second review" below, found two flaws in the prediction. Desmond approved a third freeze to fix them before M6, since M6 has not run and the prediction still comes before any result. Two files changed. `docs/m5_prediction.md` got the fixes, and `scripts/baseline.py` now records the Python, PyTorch and Transformers versions, the platform and the thread count in `baseline.json`. The other 23 hashes and the run settings are unchanged, and the rendered prompt is byte for byte the same. The baseline rerun against the third lock gave MiniLM predictions with the same SHA-256 on all three sets, and identical scores, gate and per-kind tables.

A comparison of our prompt with the paper's found that the paper's final prompt, `exp8_math/runs/p50c/message.txt`, holds an "EXPLAIN EVERY STEP" paragraph that its first-turn prompt, the one the brief adapted, does not. Desmond approved a fourth freeze to add it. The paragraph asks for 3 to 8 sentences before every block and a summary of the path to the edits in the final answer. Two files changed, the template `src/wsp/harness/task_prompt.txt` and the rendered `docs/m5_task_prompt.txt`. The paragraph matches the paper's character for character. The run settings are unchanged, and the baseline rerun gave the same prediction hashes and identical scores again. The paper also sent back a block that came without an explanation, at most twice in a row. That check is still unbuilt, as the M4 report lists, so the paragraph is an instruction only.

## Baseline

[scripts/baseline.py](../scripts/baseline.py) checks the lock, runs the unedited MiniLM on all three sets, and grades each with `wsp.score.grade`. It writes `runs/base/baseline.json` and MiniLM's predicted spans for every row, `runs/base/pred_<set>.jsonl`, which M7 needs for the paired comparison. The run takes 2 minutes on CPU.

The predictions are reproducible on this machine, a MacBook Pro with an M4 Pro. Three runs of the main eval gave the same spans on all 5,004 rows: with 1 PyTorch thread, with 14, and on the Apple GPU through MPS. The GPU took 5.4 s against 10.5 s on CPU. Scoring stays on CPU anyway. The saving is a few minutes in a run of hours, and moving to the GPU would change `predict.py` and the server, both locked, for a guarantee only measured on one set. Another machine may differ in the last bits of a logit and flip a near-tie, so M7 should score on this machine and compare its environment with the one `baseline.json` records.

This is the base row of the main table from §5 of the brief. The ± values are bootstrap standard errors over rows.

| Model | Singapore F1 | Nemotron F1, all labels | Mechanism metric | Weights changed |
|---|---|---|---|---|
| Base | 0.674 ± 0.004 | 0.952 ± 0.001 | chosen at M7 | none |

Per label, on the three scored labels:

| Label | Singapore F1 | Precision | Recall | Nemotron F1 |
|---|---|---|---|---|
| `first_name` | 0.580 | 0.691 | 0.499 | 0.985 |
| `last_name` | 0.743 | 0.657 | 0.856 | 0.984 |
| `national_id` | 0.737 | 0.953 | 0.602 | 0.959 |
| All three | 0.674 | 0.684 | 0.664 | 0.984 |

![Baseline F1 of the unedited MiniLM on each scored label, Singapore against Nemotron: first_name 0.580 against 0.985, last_name 0.743 against 0.984, and national_id 0.737 against 0.959. Pooled over the three labels it scores 0.674 against 0.984, the pair the gate compares](figures/baseline_f1.svg)

This figure draws the per-label table. Each label has a pair of bars, Singapore in red above Nemotron in grey. MiniLM trails on all three labels, and most on `first_name`. The pair below the line pools the three labels, and it is the pair the gate compares. Pooled F1 counts every entity together rather than averaging the three labels. `first_name` has the most entities, so it pulls the pooled score to 0.674, below the mean of the three, 0.687.

On Singapore text, MiniLM makes 5,555 errors. 3,051 are the wrong type, 689 are missed entities, 619 are wrong boundaries, and 1,196 are false positives.

On the practice set the baseline gives F1 0.682, the same value M3 reported. The frozen pipeline reproduces the earlier number.

## The gate passes

The gate asks whether MiniLM does worse on Singapore text than on the text it was trained on, by more than noise. On the three scored labels, Singapore F1 is 0.674 and Nemotron F1 is 0.984. The gap is 0.310.

| Reading of "its standard error" | Standard error | Twice it | Passes |
|---|---|---|---|
| Singapore F1 | 0.0042 | 0.0083 | yes |
| The gap, from both sets' standard errors | 0.0044 | 0.0087 | yes |

The gap is 36 times the stricter threshold, so the reading does not matter.

## Where MiniLM fails

[src/wsp/outcomes.py](../src/wsp/outcomes.py) records what happened to each gold entity and sorts it by kind. On practice it reproduces M3's numbers exactly: 134 of 159 NRICs right with tax context, 0 of 61 without. Its count of right entities equals the grader's true positives, 1,738 on practice.

F1 says how far each label falls short. It does not say which entities MiniLM gets wrong, and the prediction and the edit work at that level. So this section splits each label's entities in the main eval, the Singapore test set, into kinds. A kind is a group of entities that MiniLM might handle differently, such as NRICs with and without the word "tax" before them, or the first and later words of a name.

![Baseline share of Singapore test entities the unedited MiniLM gets exactly right, by kind. NRICs without tax context are 0% right, most tagged biometric_identifier or missed. Later words of given names are 5% right, 1,227 of 1,434 tagged last_name. The other kinds are 62% to 89% right](figures/baseline_kinds.svg)

Each bar is the share of one kind's entities that MiniLM tags exactly right, with the exact span and the right label. The bars are grouped under the label they belong to. The figure uses this share rather than F1 because a false positive has no gold entity, so it belongs to no kind. The misses are not spread evenly. Two kinds are almost never right, and the note on each says what MiniLM outputs instead. The bar outlined in green is the one kind the prediction expects the edit to fix.

The table gives the counts and the most common errors on the main eval:

| Kind | Entities | Right | Most common errors |
|---|---|---|---|
| NRIC, no tax context | 323 | 0% | `biometric_identifier` 168, missed 118 |
| NRIC, tax context | 711 | 87% | `biometric_identifier` 37, boundary 24 |
| Given name of several words, later word | 1,434 | 5% | `last_name` 1,227 |
| Given name of one word | 3,376 | 62% | `last_name` 766, missed 212 |
| Given name of several words, first word | 1,434 | 67% | `last_name` 294, missed 69 |
| Surname of one word | 2,688 | 89% | missed 78, `first_name` 77 |
| Surname of several words, first word | 1,274 | 81% | `first_name` 72, `company_name` boundary 57 |
| Surname of several words, later word | 1,734 | 83% | missed 117, `company_name` boundary 90 |

Two findings matter for M6 and M7.

1. **The biggest single failure is a labelling convention.** openpii marks names like "Enie Idilia" as one given name. MiniLM types the second word as `last_name` 1,227 times out of 1,434. Nemotron almost never has a given name of more than one word, so this gap comes from the two datasets' conventions, not from anything about Singapore.
2. **The NRIC fix has a small ceiling on overall F1.** If an edit got all 323 NRICs without tax context right and changed nothing else, Singapore F1 would rise by 0.017. That is about four times the Singapore standard error, so a full fix is detectable but small. The mechanism metric in M7 has to carry most of the NRIC result.

## Prediction

[docs/m5_prediction.md](m5_prediction.md) records the prediction for each kind before the run. It predicts that the edit fixes NRICs without tax context and leaves every name kind unchanged. Overall Singapore F1 rises by 0.002 to 0.011, from 33 to 205 more NRICs right. The Nemotron scores stay within 0.005 of the base. It also says which results would contradict the paper's pattern. The prediction has its own commit on `main`, made before M6, so its timestamp shows it came before the run.

## Failing rows

[docs/m5_failing_rows.md](m5_failing_rows.md) shows 12 main-eval rows, generated by [scripts/failing_rows.py](../scripts/failing_rows.py) with a fixed seed. Rows 1 to 8 each show one failure: an NRIC with and without tax context, a given name typed as a surname, misses, a boundary error, and a false positive. Rows 9 to 12 hold the most failures among the rest. MiniLM gets 10 of 47 scored entities right on these rows.

Two patterns show up that no table above isolates:

- MiniLM tags titles such as `Mstr`, `Miss` and `Judge` as `first_name`. These count as false positives.
- MiniLM breaks hyphenated names such as `Ionel-Florin` and `Marie-France` at the hyphen, so they score as boundary errors.

## Cost estimate

[scripts/estimate_cost.py](../scripts/estimate_cost.py) extrapolates from dry run 2. It leaves out dry run 1, whose per-call `cost` field holds the CLI's running total, the logging bug M4 fixed. A least-squares fit of dry run 2's per-call costs to its token counts recovers the prices exactly, with a largest error of 4 × 10⁻¹⁶ dollars. The prices are $4 per million input tokens, $8 to write the cache, $0.20 to read it, and $20 for output.

Each turn in dry run 2 added about 3,800 tokens of context and 2,900 output tokens. Every turn rereads the whole context from the cache, so cost grows with the square of the turn count. The paper's only 50-pass run, `p50c`, took 158 turns. Its 10-pass run took 74.

| Scenario | Turns | Cost | Wall time | Context at the end |
|---|---|---|---|---|
| As long as the paper's 10-pass run | 74 | $8.67 | 0.8 h | 284,000 tokens |
| As long as the paper's 50-pass run | 158 | $23.58 | 1.6 h | 605,000 tokens |
| Long | 250 | $46.09 | 2.5 h | 956,000 tokens |

These numbers are estimates, and each rests on an assumption that M6 will test:

- The wall time is a floor. It counts 35 s per reply, the dry-run mean. Code runs in dry run 2 took 0.04 s on average and its one forward pass took 6 s. Real analysis code and forward passes with activations will take longer. The paper's `p50c` took 20.3 hours, but most of that was Qwen generating text, which MiniLM does not do.
- The estimate assumes the context grows without limit. If the CLI compacts the conversation, the reread context shrinks, but the new context is written to the cache at $8 per million tokens. The cost is roughly an upper bound for a given turn count. The context window of Opus 5.5 is not recorded anywhere in this project, so whether 605,000 tokens fits is unmeasured.
- The cost is the API-price equivalent the CLI reports. The run uses the claude.ai subscription, so the real limit is the subscription's usage cap, not dollars.

## Show me

- The baseline table: the main table and per-label tables above.
- Per-category scores: the "Where MiniLM fails" table.
- 12 failing rows: [docs/m5_failing_rows.md](m5_failing_rows.md).
- The prediction: [docs/m5_prediction.md](m5_prediction.md).
- The cost estimate: the table above.

## Reproduce M5

Run these commands from the repository root.

1. Run the tests.

	```bash
	uv run python -I -m unittest discover -s tests
	```

2. Freeze. This step runs once and refuses if `lock.json` exists.

	```bash
	uv run python -I scripts/freeze.py
	```

3. Score the unedited MiniLM and run the gate. This writes `runs/base/`.

	```bash
	uv run python -I scripts/baseline.py
	```

4. Render the failing rows.

	```bash
	uv run python -I scripts/failing_rows.py
	```

5. Estimate the run's cost.

	```bash
	uv run python -I scripts/estimate_cost.py
	```

6. Regenerate the figures.

	```bash
	uv run python -I docs/figures/report_figures.py
	```

## Independent review

A separate Opus 5.5 subagent reviewed the M5 code. It had the brief and the earlier reports as context, but not this session's reasoning. It found no bug in the scoring, the gate or the prediction numbers. It compared `wsp.outcomes` and the failing-rows false positives with `wsp.score.grade` on all 6,004 Singapore rows and found no row that disagreed. It made five findings, and M5 fixed all five.

| Finding | Severity | Action |
|---|---|---|
| The lock leaves out the run settings, such as effort and the system prompt | High | Fixed. The lock records them and `run_editor.py` checks them |
| Nothing protects `lock.json` itself, which is untracked | Medium | Fixed. `lock.json` and the prediction are committed. M7 still has to compare the lock hash with `baseline.json` |
| The prediction's categories and the prediction are outside the lock | Medium | Fixed. They are the `baseline` group, and `baseline.json` records the hash of each `pred_<set>.jsonl` file of MiniLM's spans |
| `check_locked_run` uses `assert` and can wipe `work/` on a rerun | Low | Fixed. It raises errors and checks the run folder first |
| The wall time is a floor, and compaction rewrites the cache | Low | Fixed in the estimate's wording |

### Second review

After the second lock was committed, Desmond asked for a review of M5. Opus 5.5 reviewed it in the main session, and a Fable 5.1 subagent reviewed it separately without that session's conclusions. Both recomputed every number in this report and the prediction from `runs/base/` and found them right. The findings:

| Finding | Severity | Action |
|---|---|---|
| The prediction's regression rule, two paired standard errors, flags 10 wrong entities out of 146,951 as a change | Medium | Fixed in the third lock. Unchanged now means within 0.005 |
| The prediction's F1 range, 0.005 to 0.015, did not follow from its own claims | Medium | Fixed in the third lock. It is now 0.002 to 0.011 |
| 917 ids in `nemotron_eval.jsonl` appear on two rows. Nemotron wrote each template once with a US persona and once with an international one under the same id | Medium | Open for M7, below. The baseline pairs rows by position, so its scores are right |
| `run_editor.py` checks the settings only when the run is named `p50` | Medium | Open for M7, below |
| The run folder leaves out five records an audit would need | Medium | Fixed in the driver, outside the lock. See "Records added to the run folder" below |
| The test suite shares sandbox folders, so two test runs at once can fail | Low | Not fixed. Run one test process at a time |
| The cost estimate assumes every turn reads the context from the cache | Low | Not fixed. A slow code run could let the cache expire, which costs a few dollars per miss |

### Records added to the run folder

M4 built the run folder to the list in §2 of the brief, which summarises the paper's logs and leaves out two of the paper's files: the raw event stream of each turn and Opus 5.5's own notes. M4 also never reads `work/` back, because Opus 5.5's code can plant symlinks there. The dry runs made no real edits, so the gaps did not show. The driver now writes five more records:

| Record | Where | Why |
|---|---|---|
| The edit each forward pass ran | `forward/<k>/edit_manifest.json`, and `forward/<k>/edit.npz` when the edit changed since the last pass. The manifest names the file that holds its diff | A mid-run result can be checked against the weights that produced it. An `original` pass records no edit |
| The CLI version and the lock hash | `run.json`, `cli_version` and `lock_sha256` | M7 can tie the run to the locked settings |
| The CLI's own session file | `cli_session.jsonl` | It holds the served model id and the stop reason of every reply. The CLI keeps it outside the project |
| Opus 5.5's small files from `work/` | `work_notes/`, with what was left out and why in `run.json` | `notes.md` and Opus 5.5's other files survive a rerun. Files over 1 MB, symlinks and hard links are left out, and the copy never follows a link |
| A refusal to start `p50` from uncommitted code | `scripts/run_editor.py` | `run.json`'s git commit then names the exact code that ran |

Three tests in [tests/test_harness.py](../tests/test_harness.py) cover them: the per-pass edits rebuild to the hashes in their manifests, only small plain files are copied, and the session file is copied. The refusal was checked on the uncommitted tree.

### Dry run 3

A short run with Opus 5.5 tested the new records and the EXPLAIN EVERY STEP paragraph against the real CLI, from a clean tree at `a2c4ebf`. It used the locked prompt with a budget of 2 and a cap of 12 turns, under the name `dry3`, so the lock's run checks did not apply. It took 8 minutes and cost $1.43. Opus 5.5 spent the 12 turns diagnosing. It ran 10 code blocks and one forward pass, on the unedited MiniLM, then submitted no edit.

- **Every record was written.** `run.json` holds the CLI version, 2.1.292, and the lock hash, which matches `lock.json`. `cli_session.jsonl` was copied, and it shows every reply came from `claude-opus-5-5` and ended with `end_turn`. `work_notes/` holds Opus 5.5's four files, and `notes.md` matches the sandbox copy byte for byte. The one forward pass records that it ran no edit. The git record says the tree was clean.
- **Opus 5.5 followed EXPLAIN EVERY STEP.** Ten of its 11 replies with a block opened with 3 to 8 sentences, and one with 9. Each said what the last result showed, what it concluded, and what it would do next and why. About half also said what would confirm or refute a hypothesis. The others were setup steps, and one of them says "Nothing is tested here". The final answer summarises the path in sections, "What I did" and "What I would have done next". Nothing suggests the send-back check is needed.
- **The forward request costs turns.** The ```forward block must hold its item ids as literal JSON. Opus 5.5 spent two turns printing 470 ids so it could paste them into the block. This matches the paper's format and costs no forward pass.

The run folder is in `runs/dry3/`. The sandbox, 2.1 GB, was deleted.

A turn cap was considered and dropped. The paper's runs had none: a run continued until every pass was used, with only the 300 code runs capped.

## Open items

- **M7's scoring must call `wsp.lock.check` and compare the lock hash with `baseline.json`.** The script does not exist yet.
- **M7 must pair Nemotron rows by position, never by id.** Joining `pred_nemotron_eval.jsonl` by id drops the regression F1 from 0.952 to 0.905. M7 should check that both files have the same rows in the same order.
- **M7 must compare `run.json` with the locked settings, whatever the run is named.** It should also refuse a run whose `run.json` says the git tree was dirty.
- **M7 should check each forward pass's edit.** Rebuild `forward/<k>/edit.npz` with `apply_npz` and compare the hashes with `forward/<k>/edit_manifest.json`.
- **A `claude` update blocks M6.** The lock records CLI version 2.1.292. If the CLI updates itself before M6, `run_editor.py` refuses to start, and the M4 context check has to be repeated on the new version before the lock is renewed.
- **The restart is still unbuilt, and the paper's run needed it.** The paper restarted its conversation every 40 turns, and `p50c` used 4 conversations. At 158 turns this run's context would reach about 605,000 tokens. If that exceeds the window, the CLI compacts the conversation instead, and Opus 5.5 does not see `notes.md` again as the prompt promises. `run.json` records whether compaction happened. The restart is a driver change outside the lock.
- **The M4 open items about usage limits and daemonized children are unchanged.**

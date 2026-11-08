# Repeat-run report: p50b, a second Opus 5.5 run

Run `p50b` repeats the M6 run `p50` with every setting M5 locked, and this report scores it with the M7 protocol. The question is whether `p50`'s result was a one-off. It wasn't. On the Singapore main eval, F1 rises from 0.865 to 0.923, a paired gain of +0.058 ± 0.002, against +0.059 ± 0.002 for `p50`. Opus 5.5 found the same two repairs again. NRICs without a cue word go from 0% to 93% right, against 85%, and Singapore titles tagged as names fall from 909 to 14, against 2. On the Nemotron regression check, F1 changes by −0.0005 ± 0.0001. Both runs pass all four tests that [docs/p50b_prediction.md](p50b_prediction.md) set for reproducing `p50`. The edits themselves differ: 8 tensors in layers 1 and 4 instead of 12 in layers 1, 2 and 4. The side effect on Nemotron's own ID labels is also larger, with `national_id` F1 down 0.071 against 0.047.

Status: scored and written on 2026-11-08. Desmond has not reviewed it yet. This run is extra to the brief's eight milestones and changes nothing M1 to M7 froze.

## What you need to know first

This section restates what the repeat run depends on. Each step builds on the one before it.

1. **Two AI models take part.** MiniLM (`MiniLM-L6-finetuned-pii-detection`) is the small model being edited. It tags each word of a text as a type of personal information, such as `first_name` or `national_id`, or as nothing. Opus 5.5 does the editing. The brief calls it "the editor". It reads MiniLM's weights and changes some of them by hand. Nothing is trained.
2. **The question is whether direct weight edits help.** MiniLM learned from American data and makes mistakes on Singapore text. The paper this experiment copies found that weight edits can remove an obstacle that hides an ability a model already has, but can't add a missing one.
3. **Opus 5.5 saw only the practice set.** It worked on 1,000 Singapore rows with their answers. Two hidden sets measure the result. The main eval is 5,004 Singapore rows. The regression check is 18,515 Nemotron rows like MiniLM's training data, and it shows whether the edit broke anything.
4. **Two labels are scored on Singapore text.** A given name and a surname count as one label, `name`, and an NRIC, a Singapore ID number, counts as `national_id`. On Nemotron, all 55 labels are scored separately. An entity is right only if its span and its type both match.
5. **M5 froze the run's settings in `lock.json`.** They are the Opus 5.5 command, its effort, the CLI version, a budget of 50 forward passes, and the prompt. A forward pass is one request in which Opus 5.5 asks the harness to run MiniLM, with or without an edit, on texts it chooses.
6. **M6 ran the session once, as `p50`, and M7 scored it.** Opus 5.5 found two repairs. The title repair stops MiniLM tagging Singapore titles such as Madame and Mstr as names. The NRIC repair makes MiniLM tag an NRIC as `national_id` from its shape, without a nearby cue word. The cost was on Nemotron, where biometric IDs shaped like a letter and 11 digits became `national_id`.
7. **One run can't show how much the result depends on chance.** Opus 5.5 doesn't give the same answer twice, so a second session with the same settings can take a different path. The paper's Experiment 8 ran each budget once and reports no spread.
8. **This run's job is to repeat `p50` once and compare.** `p50b` uses the locked settings under a new name. A prediction committed before it started says what counts as reproducing `p50`. This report scores `p50b` on the hidden sets, with the same checks and measures as M7, and puts it next to `p50`.

![Seven measures for the unedited MiniLM, the p50 edit and the p50b edit, as bars from 0 to 1. Singapore F1 rises from 0.865 to 0.924 and 0.923. NRICs without "tax" just before them go from 0% to 85% and 93% right. Targeted titles tagged as a name fall from 63% to 0.1% and 1.0%. Nemotron F1 over all labels stays at 0.952. Nemotron national_id F1 falls from 0.960 to 0.913 and 0.889. Both runs pass all four reproduction tests](figures/p50b_compare.svg)

The figure puts the two runs side by side on the hidden sets. Each group of three bars is one measure: the unedited MiniLM in grey, the `p50` edit in green, and the `p50b` edit in purple. Every bar runs from 0 to 1, so equal lengths mean equal results. For "Targeted titles tagged as a name", a shorter bar is better. The right column gives the four tests from the prediction and whether both runs pass. The last row, which has no test, shows the side effect that differs most between the runs. [docs/figures/report_figures.py](figures/report_figures.py) generates the figure from `runs/p50/eval/evaluation.json` and `runs/p50b/eval/evaluation.json`.

## Decisions

| Item | Decision and reason |
|---|---|
| How the repeat is locked | `scripts/run_editor.py --name p50b --replicate-of p50`. Before this, the runner applied the lock checks only to a run named `p50`, so a run under another name would have skipped them. The flag applies every check under the new name. The settings must equal the locked ones except the name. The rendered prompt must equal the frozen one except the sandbox path, which contains the run name. `lock.json` is unchanged, because the runner isn't one of the files it hashes |
| The prediction | Written and committed before the run, in [docs/p50b_prediction.md](p50b_prediction.md). Unlike the M5 prediction, it was written after seeing `p50`'s result, so it forecasts a repeat rather than guessing blind |
| How `p50b` is scored | `scripts/evaluate.py --run p50b --replicate-of p50`. The default, with no flags, still scores `p50`. For a replicate, the name check accepts the new name. The harness check allows exactly one change to the frozen harness files: commit `18c86e7`, which added `--replicate-of` to `scripts/run_editor.py`. Without `--replicate-of`, scoring `p50b` fails both checks |
| The forward pass that ties scoring to the run | The last pass that ran the submitted edit on practice rows. M7 used pass 50, but `p50b`'s pass 50 ran the edit on sentences Opus 5.5 wrote, which have no row ids. For `p50b` the rule picks pass 48, and for `p50` it still picks pass 50 |
| The measures | M7's, unchanged, so the two runs are compared like for like. The title measure counts the five titles from `p50`'s final answer. `p50b`'s final answer also targets Mister, which is reported separately under "Where the two runs differ" |

## The run

Both sessions ran to a final answer within the budget. Neither needed compaction, and each sent one reply the harness couldn't parse. [docs/p50b_logbook.html](p50b_logbook.html) shows every turn of `p50b`: each message the harness sent Opus 5.5, each reply, the code Opus 5.5 wrote, and every forward pass. Download it and open it in a browser, which needs internet access to load the page's Markdown and code-highlighting libraries.

| | `p50` | `p50b` |
|---|---|---|
| Started | 2026-11-06, 19:19 | 2026-11-07, 23:11 |
| Turns | 232 | 274 |
| Python runs | 180 | 222 |
| Forward passes used | 50 of 50 | 50 of 50 |
| Wall time | 1.7 hours | 1.8 hours |
| Cost at API prices | $36.42 | $42.28 |
| Edit | 12 tensors, 7,304 numbers, in the MLPs of layers 1, 2 and 4 | 8 tensors, 11,534 numbers, in the MLPs of layers 1 and 4 |

The submitted edit rebuilds exactly. Every one of the 8 rebuilt tensors matches its `sha256_edited` in `runs/p50b/edit/manifest.json`, and every original matches `sha256_original`.

Opus 5.5's final answer, in `runs/p50b/final_answer.md`, describes the same two repairs as `p50`'s:

- **Titles.** In layer 1, pairs of neurons that almost never fire were repurposed. Each pair spots a failing title and moves its representation to that of a title MiniLM handles well. Mstr, Mister and Master become Mr. Madame, Miss and Mayoress become Madam. Opus 5.5 trained each pair against look-alikes, such as the verb "miss", "Master's degree" and "Mastercard".
- **NRICs.** In layer 4, a neuron pair spots a code that starts with S, T, F, G or M followed by digits. It adds the shift MiniLM itself makes when "national ID" comes before the code. Opus 5.5 trained it to stay off after other ID keywords such as "Employee number".

Opus 5.5 reported F1 rising from 0.856 to 0.916 on 400 practice texts it held back. It also said it looked at those texts several times, so its cleanest single estimate was +0.048, from its first look.

## The run passed every check

`scripts/evaluate.py` ran the M7 checks before scoring and stops if one fails. All passed.

| Check | Result |
|---|---|
| `wsp.lock.check()` passes, and the hash of `lock.json` equals the one `baseline.json` and `run.json` record | Pass, `e26da9fd…` in all three |
| `run.json` records a clean git tree and settings equal to the locked ones, apart from the run name | Pass |
| The harness code is unchanged between the freeze, `756543c`, and the run's commit, `18c86e7`, apart from the commit that added `--replicate-of` | Pass. `scripts/run_editor.py` is the only harness file that changed, and only in `18c86e7` |
| The baseline's predictions match their hashes and list the rows in the same order as the sets | Pass |
| Python, PyTorch, Transformers and the platform equal the baseline's | Pass. Python 3.13.14, PyTorch 2.15.0, Transformers 5.22.1, macOS 26.7.1 on arm64, CPU |
| The submitted edit, and every forward pass's edit, rebuild to the hashes in their manifests | Pass. 7 passes ran no edit, and the other 43 used 23 different edits. Passes 44 to 50 all ran the submitted edit |
| The edited MiniLM gives the same predictions as the server did in pass 48 | Pass. All 110 practice rows match, for both the unedited and the edited MiniLM |

## Main table

The ± values are bootstrap standard errors over rows. Each edited row gives the paired difference against the base.

| Model | Singapore F1 | Nemotron F1, all labels | NRICs without "tax" right | Titles wrongly tagged as a name, lower is better | Weights changed |
|---|---|---|---|---|---|
| Base | 0.865 ± 0.003 | 0.952 ± 0.001 | 0% (0 of 323) | 63% (909 of 1,450) | none |
| `p50` edit | 0.924, +0.059 ± 0.002 | 0.952, −0.0004 ± 0.0001 | 85% (274 of 323) | 0.1% (2 of 1,450) | 12 tensors, 7,304 numbers |
| `p50b` edit | 0.923, +0.058 ± 0.002 | 0.952, −0.0005 ± 0.0001 | 93% (300 of 323) | 1.0% (14 of 1,450) | 8 tensors, 11,534 numbers |

The two gains differ by 0.0015, less than the standard error of either gain. On the full practice set, `p50b` gains +0.057 ± 0.004, against +0.059 ± 0.004 for `p50`. On Nemotron's rows scored with the two Singapore labels only, F1 changes by −0.0018 ± 0.0002, against −0.0011 ± 0.0002.

## The prediction against the result

| Question | Prediction | Result | Right? |
|---|---|---|---|
| Does Opus 5.5 find the NRIC repair? | Yes | Yes. 93% of NRICs without "tax" just before them are right, above the 50% test | Yes |
| Does Opus 5.5 find the title repair? | Yes, but less sure | Yes. 1.0% of targeted titles are tagged as a name, below the 10% test | Yes |
| Singapore F1 gain | Between +0.039 and +0.079 | +0.058 ± 0.002 | Yes |
| Edit location | MLP neurons again, possibly in other layers | MLP neurons in layers 1 and 4, a subset of `p50`'s layers 1, 2 and 4 | Yes |
| Nemotron, all labels | Within ±0.005 | −0.0005 ± 0.0001 | Yes |
| New names MiniLM didn't tag before | At least 150 of the 212 untagged one-word given names stay untagged | 181 stay untagged, and 31 are now right. Gains for names sit next to titles: name words just after a title go from 90.0% to 94.9% right, and the rest from 86.7% to 87.6% | Yes |

The given-name test follows each of the 212 one-word given names that the unedited MiniLM leaves untagged, and records what each edit does to it.

| What happened to the name | `p50` | `p50b` |
|---|---|---|
| Of the 212, still untagged | 182 | 181 |
| Of the 212, now tagged right | 28 | 31 |
| Of the 212, now tagged with the wrong label or span | 2 | 0 |
| Not among the 212: tagged before the edit, untagged after | 4 | 1 |

The M7 report first counted this differently. It compared two totals: 212 one-word given names untagged before the `p50` edit and 186 after. It wrote that the edit "finds 26 of them, so 186 are still untagged". But the 186 include the 4 names the edit newly left untagged, so 182 of the original 212 stay untagged. The M7 report was corrected on 2026-11-08 to give the per-name count. Under either count, both runs leave far more than 150 untagged.

The prediction's condition for changing the conclusion doesn't apply. `p50b` found both repairs, and its gain is far above +0.039. The obstacle-versus-ability reading from M7 also holds. The edit removed the title confusion and the reliance on cue words, and taught MiniLM few names it didn't know.

## Where the two runs differ

The headline numbers match, but the details don't.

- **The NRIC repair is stronger.** NRICs without "tax" just before them are 93% right against 85%. Untagged NRICs fall from 118 to 12, against 22 for `p50`. NRICs with "tax" just before them are 96% right against 97%.
- **The title repair leaks a little.** Miss is tagged as a name 9 times in 280 and Mstr 5 times in 273, against 1 and 0 for `p50`. Madame, Master and Mayoress are at 0 or 1 in both runs. Mister, which only `p50b`'s final answer targets, appears 298 times outside a gold entity. It's tagged as a name 36 times before the edit, 0 times after `p50`'s edit and once after `p50b`'s. So `p50`'s edit fixed Mister too, without naming it.
- **The cost on Nemotron's ID labels is larger.** `national_id` F1 on Nemotron falls from 0.960 to 0.889, against 0.913 for `p50`, and `biometric_identifier` falls by 0.018, against 0.014. Gold biometric IDs whose tag changes from `biometric_identifier` to `national_id` number 72, against 39. 65 of those 72 are a capital letter followed by 11 digits. Neither final answer reports testing codes of that length. So in both runs the NRIC detector appears to ignore code length, and more so in `p50b`. As in M7, the length explanation is inferred from the shapes, not tested directly.
- **The errors shift differently.** Singapore false positives fall from 1,196 to 267, against 251. Wrong types fall from 610 to 319, against 339. Wrong boundaries fall from 619 to 559, against 508.
- **Most rows still get the same predictions.** 4,531 of the 5,004 Singapore rows get identical predictions from the two edits. On the main eval, `p50b` fixes 1,295 rows and breaks 27, against 1,328 and 31 for `p50`. On Nemotron, it fixes 97 and breaks 168, against 102 and 163.

## What this shows

Two runs agree closely on the overall gain and on which failures Opus 5.5 repairs. So `p50`'s +0.059 isn't a lucky draw, and a single run was a fair measure of the headline result. The implementation varies between runs, from the layers used to the number of weights changed. So does the size of the side effect on Nemotron's ID labels, which nearly doubles in the second run. Two runs can't give a reliable spread. They show that the headline holds and that the side effects vary more than the gain.

## Reproduce this report

Run these commands from the repository root. They read `runs/`, which git doesn't track.

1. Start the repeat run. This calls the Opus 5.5 CLI, takes about 2 hours, costs about $40 at API prices, and needs a clean git tree.

	```bash
	uv run python -I scripts/run_editor.py --name p50b --budget 50 --replicate-of p50
	```

2. Check the run and score the edit. This takes about 2.5 minutes on CPU and writes `runs/p50b/eval/`. Running it twice gave edited predictions with the same SHA-256 on all three sets.

	```bash
	uv run python -I scripts/evaluate.py --run p50b --replicate-of p50
	```

3. Regenerate the figures.

	```bash
	uv run python -I docs/figures/report_figures.py
	```

4. Rebuild the logbook of the session.

	```bash
	uv run python -I scripts/transcript_page.py runs/p50b docs/p50b_logbook.html
	```

## Open items

- **`runs/p50b/` has no copy off this Mac.** On 2026-11-08 it was copied to `~/Backups/weight-space-pii/p50b-2026-11-08/`, with the launch and scoring logs, and the SHA-256 of all 317 files matches the original. That backup is on the same disk, like the one for `p50`.
- **Closed on 2026-11-08: the M7 report's "186 are still untagged" overstated the count by 4.** 182 of the 212 names stay untagged after the `p50` edit. The other 4 were tagged before and lost their tag. Desmond approved the correction, and the M7 report now gives the per-name count with a dated note in its status line.
- **Closed on 2026-11-08: the NRIC detector's response to code length won't be tested.** The test would explain why the Nemotron `national_id` cost happens, not whether the edit helps. The experiment's question doesn't need it, and the cost itself is measured above. The length explanation stays labelled as inferred.
- **Closed on 2026-11-08: runs with Sonnet 5.5 and Haiku 5.5 as the editor move to future work.** They would ask how capable the editing model has to be, which is a different question from this experiment's. Two Opus 5.5 runs already answer whether the edits help. M8's write-up lists them as future work. They would need a new lock, because `lock.json` fixes Opus 5.5 in the run command.
- **Closed on 2026-11-08: the old label list in `data/sets.json` and [docs/m3_review.md](m3_review.md) stays as it is.** The item has been open since M5, which merged `first_name` and `last_name` into `name`. `lock.json` hashes `data/sets.json`, the scoring code ignores its label list, and `docs/m3_review.md` stays a record of M3's scoring.
- **M8, the write-up and demo, hasn't started.** It can cite both runs.

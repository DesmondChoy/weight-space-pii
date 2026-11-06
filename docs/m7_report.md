# M7 report: evaluating the edited model

M7 scores MiniLM with Opus 5.5's edit on the two hidden sets, which Opus 5.5 never saw, and compares it with the unedited MiniLM and with the M5 prediction. On the Singapore main eval, F1 rises from 0.865 to 0.924, a paired gain of +0.059 ± 0.002. On the Nemotron regression check, F1 changes by −0.0004 ± 0.0001, inside the prediction's limit for unchanged. Both repairs Opus 5.5 described work on the hidden text. NRICs without a cue word go from 0% to 85% right, and Singapore titles tagged as names fall from 909 to 2. The cost is on Nemotron's own ID labels. `national_id` F1 there falls by 0.047, because biometric IDs shaped like a letter and 11 digits are now tagged `national_id`. The prediction got the NRIC category right and every name category right, but it expected an overall gain of at most 0.009. It follows §5 and §7 of [weight-space-pii-handoff.md](weight-space-pii-handoff.md), with the decisions in the M2 to M6 reports.

Status: scored and written on 2026-11-06. Desmond chose 5 fixed and 5 broken rows from each set. Waiting for Desmond to review the items under "Show me". Committed on `main`, where `git log` lists every M7 commit.

## What you need to know first

The brief and six earlier reports hold the background. This section restates what M7 depends on. Each step builds on the one before it.

1. **Two AI models take part.** MiniLM (`MiniLM-L6-finetuned-pii-detection`) is the small model being edited. It tags each word of a text as a type of personal information, such as `first_name` or `national_id`, or as nothing. Opus 5.5 does the editing. The brief calls Opus 5.5 "the editor". It reads MiniLM's weights, the numbers inside MiniLM, and changes some of them by hand. Nothing is trained.
2. **The question is whether direct weight edits help.** MiniLM learned from American data and makes mistakes on Singapore text. The paper this experiment copies found that weight edits can remove an obstacle that hides an ability a model already has, but cannot add a missing one. This experiment asks whether the same holds for MiniLM.
3. **Opus 5.5 saw only the practice set.** It worked on 1,000 Singapore rows with their answers. Two hidden sets measure the result. The main eval is 5,004 Singapore rows. The regression check is 18,515 Nemotron rows like MiniLM's training data, and it shows whether the edit broke anything.
4. **Two labels are scored on Singapore text.** A given name and a surname count as one label, `name`, and an NRIC, a Singapore ID number, counts as `national_id`. A `tax_id` tag on an NRIC also counts as `national_id`. On Nemotron, the regression check scores all 55 labels separately. An entity is right only if its span and its type both match. These rules come from M3 and M5.
5. **M5 froze everything that decides a score and wrote a prediction.** `lock.json` holds the hash of every such file. M5 scored the unedited MiniLM, the baseline, and committed [docs/m5_prediction.md](m5_prediction.md) before the run. The prediction sorts Singapore entities into eight categories. It says the edit fixes NRICs without "tax" just before them and leaves every name category unchanged, and that Singapore F1 rises by 0.002 to 0.009. A category is fixed if its share right rises by at least 10 points.
6. **M6 ran the 50-pass session once.** Opus 5.5 changed 12 tensors in the MLPs of layers 1, 2 and 4. Its final answer describes two repairs. The title repair stops MiniLM tagging Singapore titles such as Madame and Mstr as names. The NRIC repair makes MiniLM tag an NRIC as `national_id` from its shape, without a nearby cue word. Opus 5.5 reported F1 rising from 0.856 to 0.921 on half of the practice set, but that half helped choose the edit, so the number was likely a little high.
7. **M7's job is to measure the edit cleanly.** M7 checks that the run and the edit are the ones M5 locked and M6 recorded. It then scores the edited MiniLM on the hidden sets, reports the paired difference against the baseline, compares each category with the prediction, and chooses the mechanism metric. The brief defines that metric as the share of the failure Opus 5.5 targeted that the edit fixed.

![Share of Singapore main-eval entities MiniLM tags exactly right, before and after the edit, in eight categories. NRICs without "tax" just before them go from 0% to 85%, fixed as predicted. NRICs with "tax" just before them go from 87% to 97%. Every name category rises by 1 to 3 points, which counts as unchanged, as predicted](figures/m7_kinds.svg)

The first figure compares the edited MiniLM with the prediction, category by category. Each pair of bars is one category of Singapore entities, with the unedited MiniLM in grey above the edited MiniLM in green. A bar's length is the share of the category's entities tagged with the exact span and the right label. The two columns on the right give the prediction's verdict and the result under the prediction's own rule. Green text means the two agree, and they agree in all eight rows. The note at the bottom marks the narrowest call. NRICs with "tax" just before them rose 9.99 points, just under the 10 points needed to count as fixed.

![Left, Singapore main-eval errors by kind, before and after the edit. False positives fall from 1,196 to 251, and wrong types from 610 to 339. Right, the six Nemotron labels whose F1 moved most. national_id falls by 0.047 and biometric_identifier by 0.014, and all 55 labels together change by -0.0004](figures/m7_errors.svg)

The second figure shows where the gain came from and what it cost. The left half counts Singapore errors by kind. A false positive is a tag where the answer has no entity, and most of the drop there is titles no longer tagged as names. The right half shows the change in Nemotron F1 for the four labels that fell most and the two that rose most, out of 55. Bars left of the line are losses. The note under it names the cause of the largest loss. [docs/figures/report_figures.py](figures/report_figures.py) generates both figures from `runs/p50/eval/evaluation.json`.

## Decisions

| Item | Brief | Decision and reason |
|---|---|---|
| The mechanism metric | §5 chooses one after reading the final answer | Two, one for each repair the final answer describes. The NRIC metric is the share of NRICs without "tax" just before them that are right. The title metric is the share of Singapore titles that MiniLM tags as part of a name. Both were defined from the final answer before any edited score existed. A single metric would hide one repair, and the two repairs move different parts of F1 |
| The titles the metric counts | Not specified | The five titles the final answer names: Madame, Master, Miss, Mstr and Mayoress. Mr and Mrs are the control, because the final answer says MiniLM already handled them. Mdm, Dr, Mx and Madam are counted as untargeted. A title inside a gold entity is skipped |
| The Nemotron column of the main table | §5 has one Nemotron F1 | All 55 labels, the regression check, as in the M5 baseline. The score on the two Singapore labels is reported below it |
| A changed row | Not specified | A row whose errors, false positives plus missed or wrong entities, differ between the two models. Fixed means fewer errors, broken means more. A row with the same number of errors but different tags is mixed |

## The edit and the run passed every check

[scripts/evaluate.py](../scripts/evaluate.py) runs these checks before it scores anything and stops if one fails. It closes the M7 checks that the M5 and M6 reports left open. All passed.

| Check | Result |
|---|---|
| `wsp.lock.check()` passes, and the hash of `lock.json` equals the one `baseline.json` and `run.json` record | Pass, `e26da9fd…` in all three |
| `run.json` records a clean git tree and settings equal to the locked ones: the CLI command without its session ID, its environment, its version, the budget and the turn limit | Pass |
| The harness code is unchanged between the seventh freeze, `756543c`, and the run's commit, `db4aaed` | Pass. Only `docs/m5_failing_rows.md`, `docs/m5_report.md` and `scripts/failing_rows.py` changed |
| The baseline's predictions match the hashes in `baseline.json` and list the same rows in the same order as the sets | Pass. Rows are paired by position, never by id, because 917 Nemotron ids appear twice |
| Python, PyTorch, Transformers and the platform equal the ones `baseline.json` records | Pass. Python 3.13.14, PyTorch 2.15.0, Transformers 5.22.1, macOS 26.7.1 on arm64, CPU with 10 threads |
| The submitted edit rebuilds to the hashes in its manifest | Pass, 12 tensors |
| Every forward pass's edit rebuilds to the hashes in its manifest | Pass. 7 passes ran no edit, and the other 43 used 18 different edits. Passes 39 to 50 all ran the submitted edit |
| The edited MiniLM gives the same predictions as the server did in pass 50 | Pass. All 500 practice rows match, for both the unedited and the edited MiniLM |

The last check ties M7 to the run. It shows that loading the edit into MiniLM here gives what Opus 5.5 saw. Running the script twice gave edited predictions with the same SHA-256 on all three sets.

## Main table

The ± values are bootstrap standard errors over rows. The edited row gives the paired difference against the base, with both models scored on the same random redraws of rows.

| Model | Singapore F1 | Nemotron F1, all labels | NRICs without "tax" right | Titles tagged as a name | Weights changed |
|---|---|---|---|---|---|
| Base | 0.865 ± 0.003 | 0.952 ± 0.001 | 0% (0 of 323) | 63% (909 of 1,450) | none |
| Edited, 50 passes | 0.924, +0.059 ± 0.002 | 0.952, −0.0004 ± 0.0001 | 85% (274 of 323) | 0.1% (2 of 1,450) | 12 tensors, 7,304 numbers, 0.03% of MiniLM's 22.6 million |

Per label, on the two Singapore labels:

| Label | Set | Base F1 | Edited F1 | Precision | Recall |
|---|---|---|---|---|---|
| `name` | Singapore | 0.874 | 0.922 | 0.874 → 0.952 | 0.874 → 0.894 |
| `national_id` | Singapore | 0.737 | 0.950 | 0.953 → 0.965 | 0.602 → 0.935 |
| Both | Singapore | 0.865 | 0.924 | 0.878 → 0.953 | 0.852 → 0.897 |
| Both | Nemotron | 0.988 | 0.987, −0.0011 ± 0.0002 | 0.989 → 0.986 | 0.988 → 0.988 |

The gain holds on the full practice set too, +0.059 ± 0.004, and Opus 5.5's own estimate on its held-back half was +0.065 ± 0.005. So tuning on the held-back half raised Opus 5.5's number only a little. The main eval gives the clean result.

## The two repairs, measured

**The NRIC repair works on hidden text.** Of the 323 NRICs without "tax" just before them, the edited MiniLM gets 274 right, against none before. It also fixes most of the ones MiniLM missed entirely. Untagged NRICs fall from 118 to 22, and NRICs tagged `biometric_identifier` from 168 to 6. NRICs with "tax" just before them rise from 87% to 97% right, 622 to 693 of 711.

**The title repair works on hidden text, and only on the targeted titles.** Before the edit, MiniLM tagged 909 of the 1,450 targeted titles as part of a name. After it, MiniLM tags 2.

| Title | Occurrences | Tagged as a name, base | Tagged as a name, edited |
|---|---|---|---|
| Madame | 330 | 42% | 0.3% |
| Master | 304 | 25% | 0% |
| Miss | 280 | 72% | 0.4% |
| Mstr | 273 | 90% | 0% |
| Mayoress | 263 | 94% | 0% |
| Mr and Mrs, the control | 560 | 4 | 4 |
| Dr, Mx and Madam, untargeted | 39 | 5 | 5 |

The control and untargeted titles are tagged exactly as before. No row of the main eval holds "Mdm", the title the final answer says it could not handle.

**Names after a title also improve.** Of the 1,372 name words that directly follow a targeted title, 87% were right before and 95% after. This is the second half of the title repair, where "Master" no longer makes the next name read as a company.

## The prediction against the result

**Every category verdict matched.** The one category predicted fixed was fixed, and the seven predicted unchanged stayed within 10 points. The figure at the top gives each pair.

| Category | Entities | Base | Edited | Change | Predicted | Result |
|---|---|---|---|---|---|---|
| NRIC without "tax" just before it | 323 | 0% | 85% | +84.8 | Fixed | Fixed |
| NRIC with "tax" just before it | 711 | 87% | 97% | +10.0 (9.99) | Unchanged | Unchanged |
| Given name of one word | 3,376 | 85% | 86% | +1.8 | Unchanged | Unchanged |
| Given name of several words, first word | 1,434 | 87% | 90% | +2.8 | Unchanged | Unchanged |
| Given name of several words, later words | 1,434 | 91% | 92% | +1.4 | Unchanged | Unchanged |
| Surname of one word | 2,688 | 92% | 94% | +1.2 | Unchanged | Unchanged |
| Surname of several words, first word | 1,274 | 87% | 90% | +3.1 | Unchanged | Unchanged |
| Surname of several words, later words | 1,734 | 83% | 86% | +2.6 | Unchanged | Unchanged |

**The overall Singapore number was wrong by a factor of about seven.** The prediction expected F1 to rise by 0.002 to 0.009. It rose by 0.059. Two things the prediction did not foresee account for most of it.

1. **The title repair removes false positives, which no category counts.** A category counts the answer's entities, and a title tagged as a name is not one. Singapore false positives fall from 1,196 to 251, and `name` precision rises from 0.874 to 0.952. The M6 report flagged this before scoring.
2. **The NRIC repair fixed more NRICs than the prediction allowed.** The prediction assumed the 118 NRICs MiniLM did not tag at all would stay untagged, which capped the fix at 205. The edit fixed 274, including most of the untagged ones.

**The prediction's test for new knowledge does not fire.** The prediction said it would change its mind if the edit made MiniLM find names it misses, such as many of the 212 one-word given names it does not tag. The edit finds 26 of them, so 186 are still untagged. Across all name words, the gains sit mostly next to titles. Name words in an entity just after a title, 2,531 of them, go from 90.0% to 94.7% right. The other 9,409 go from 86.7% to 88.0%. Most of the names fixed away from a title were already tagged but as a company name or with the wrong boundary. Only 35 were names MiniLM did not tag before. This fits the paper's pattern. The edit removed obstacles, the title confusion and the reliance on cue words, and added no names MiniLM did not know. M8 discusses this further. That the remaining 35 come from other causes is a guess, which this report does not test.

## The cost on Nemotron

Both Nemotron scores count as unchanged under the prediction's rule. F1 on all 55 labels changes by −0.0004 ± 0.0001, and F1 on the two Singapore labels by −0.0011 ± 0.0002. Each change is more than twice its standard error, so it is real, but both are under the rule's limit of 0.005.

The loss is concentrated in the labels the prediction named as most at risk.

| Label | Base F1 | Edited F1 | Change |
|---|---|---|---|
| `national_id` | 0.960 | 0.913 | −0.047 |
| `biometric_identifier` | 0.996 | 0.983 | −0.014 |
| `certificate_license_number` | 0.975 | 0.972 | −0.003 |
| `vehicle_identifier` | 0.986 | 0.983 | −0.003 |
| `health_plan_beneficiary_number` | 0.988 | 0.987 | −0.001 |
| `unique_id` | 0.925 | 0.928 | +0.002 |

The edited MiniLM still gets Nemotron's own national IDs right, 527 against 526 before. The loss is 57 new false `national_id` tags. 46 of them sit on biometric IDs. 35 of those are a capital letter followed by 11 digits, such as `F98324157683`, and in the examples read, most follow the words "biometric identifier". Another 7 are shaped `AAA-9999999999`, and 4 are a letter and 6 to 9 digits. An NRIC is a letter, 7 digits and a letter. The final answer reports tests on codes of a letter and 7 or 8 digits, but not on longer ones. So the NRIC detector responds to a letter followed by digits without checking the length, and Opus 5.5's checks did not cover the length that Nemotron's biometric IDs use. That the detector ignores length is inferred from these shapes, not tested directly.

## Changed rows

| Set | Fixed | Broken | Mixed |
|---|---|---|---|
| Singapore main eval, 5,004 rows | 1,328 | 31 | 626 |
| Nemotron regression check, 18,515 rows | 102 | 163 | 67 |

`runs/p50/eval/changed_rows.jsonl` lists every changed row with its errors before and after and the tags that differ. Desmond asked for 5 fixed and 5 broken rows from each set. [scripts/changed_rows.py](../scripts/changed_rows.py) draws them at random with a fixed seed and renders them in [docs/m7_changed_rows.md](m7_changed_rows.md), each with every tag that changed and the gold label on the same text.

- **All five fixed Singapore rows involve a targeted title.** In four, MiniLM stops tagging Mstr, Madame or Miss as a name. In the fifth, the name after "Master" goes from untagged to tagged. Two of the rows also show an NRIC after "Tax" switching from `tax_id` to `national_id`, which was already scored right.
- **Three of the five broken Singapore rows sit next to a repair.** In row 6, a 16-digit card number is now tagged `national_id`, a false positive. That is one row, and the Singapore `national_id` false positives rise only from 31 to 35 overall. In row 7, "Master" is no longer tagged, but the name after it, "M'halla Avshalom", is split differently and still wrong. In row 9, the second word of the name after "Madame" loses its tag. Rows 8 and 10 each lose one name word, to `company_name` and to `user_name`, with no title or NRIC nearby.
- **The Nemotron rows are mostly single tags flipping in labels the repairs do not target.** Dates, times, an email, occupations and a fax number change in both the fixed and the broken rows. One broken row, 16, shows the ID cost from the section above. In `BIO-7834561290`, the edited MiniLM tags "BIO" as `national_id` and the digits as a separate biometric ID. That these flips come from small shifts in labels that nearly tied before the edit is a guess, which this report does not test.

## Show me

- The main table, under "Main table".
- The per-category results next to the prediction, in the first figure and under "The prediction against the result".
- The rows the edit changed, both fixed and broken: 5 of each from each set, in [docs/m7_changed_rows.md](m7_changed_rows.md).

## Reproduce M7

Run these commands from the repository root. Both read `runs/`, which git does not track.

1. Check the run and the edit, then score the edited MiniLM. This takes about 2.5 minutes on CPU and writes `runs/p50/eval/`.

	```bash
	uv run python -I scripts/evaluate.py
	```

2. Draw and render the changed rows.

	```bash
	uv run python -I scripts/changed_rows.py
	```

3. Regenerate the figures.

	```bash
	uv run python -I docs/figures/report_figures.py
	```

## Open items

- **`runs/p50/` has no copy off this Mac.** The backup in `~/Backups/weight-space-pii/p50-2026-11-06/` is on the same disk. It does not yet hold `runs/p50/eval/`, which `scripts/evaluate.py` can rebuild.
- **The NRIC detector's response to IDs of other lengths is inferred, not tested.** A direct test would run the edited MiniLM on the same sentence with codes of a letter and 7 to 12 digits. It is a question for the write-up, not a change to the edit, which is frozen.
- **`data/sets.json` still lists `first_name` and `last_name` as scored labels, and [docs/m3_review.md](m3_review.md) predates the merge.** Both are unchanged from the M5 report.

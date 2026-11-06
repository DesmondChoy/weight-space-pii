# M5 prediction for the 50-pass run

This is what we expect Opus 5.5's edit to do, written before the M6 run as §4 of the [brief](weight-space-pii-handoff.md) requires. M7 compares the edited MiniLM with it. The starting numbers are the unedited MiniLM's scores in `runs/base/baseline.json`, which `scripts/baseline.py` wrote from the locked sets.

The brief asks for this file in the run folder. `runs/` is not in git, and the driver refuses to start a run whose folder already exists, so the committed copy lives here. M6 copies it into `runs/p50/` after the run.

On Singapore text, a word tagged `first_name` or `last_name` counts as right on any name word, because given names and surnames are scored as one category, `name`. On the Nemotron regression check, every label is scored separately, so there the two stay apart.

## What "fixed" means

A category counts as **fixed** if the share of its entities that MiniLM gets right rises by at least 10 percentage points on the main eval. It counts as **unchanged** if the share moves by less than that. A fall of 10 points or more counts as **broken**.

For the Nemotron scores, **unchanged** means the edited F1 is no more than 0.005 below or above the base F1. The score counts as **worse** only if it falls by more than 0.005 and by more than two paired bootstrap standard errors, so neither noise nor a negligible shift can trigger it. **Better** is the same rule upward. M7 reports the paired difference and its standard error either way.

## Prediction

Each share is the share of entities right on the main eval, 5,004 rows. "Just before" means within the 60 characters before the number, about ten words. Every prediction is a guess made before the run.

| Category | Entities | Base share right | Prediction | Reason |
|---|---|---|---|---|
| NRIC without "tax" just before it | 323 | 0% | Fixed | MiniLM marks the number but picks the wrong ID type: 168 of the 323 come out as `biometric_identifier`. That is an obstacle hiding an ability, like the paper's looping case. The 118 NRICs MiniLM does not mark at all stay missed. |
| NRIC with "tax" just before it | 711 | 87% | Unchanged | MiniLM already reads these as `tax_id`, which counts as right. An edit that pushes NRICs toward `national_id` still counts as right. |
| Given name of one word | 3,376 | 84% | Unchanged | Most errors are names MiniLM does not tag at all (212) or tags as part of a longer stretch. Recognizing a name it has never seen needs knowledge MiniLM lacks, which the paper found editing cannot add. |
| Given name of several words, first word | 1,434 | 87% | Unchanged | Same as above. |
| Given name of several words, later words | 1,434 | 91% | Unchanged | Same as above. |
| Surname of one word | 2,688 | 92% | Unchanged | Already mostly right. |
| Surname of several words, first word | 1,274 | 87% | Unchanged | Errors split between misses and names tagged as part of a company name. No single cause to target. |
| Surname of several words, later words | 1,734 | 83% | Unchanged | Same as above. |

## Overall scores

| Score | Base | Prediction |
|---|---|---|
| Singapore F1, main eval | 0.865 ± 0.003 | Rises by 0.002 to 0.009. Each NRIC without "tax" just before it that becomes right adds about 0.00004. "Fixed" needs at least 33 of the 323 right, which adds 0.0015. The 118 that MiniLM does not mark stay missed, so at most 205 can be fixed, which adds 0.0090. The range assumes every other category stays where it is. |
| Nemotron F1, the two Singapore categories | 0.988 ± 0.001 | Unchanged |
| Nemotron F1, all labels (regression check) | 0.952 ± 0.001 | Unchanged. The labels most at risk are the ones an NRIC edit could shift: `national_id` 0.960, `tax_id` 0.967, `biometric_identifier` 0.996, `health_plan_beneficiary_number` 0.988. |

## What would change my mind

If Opus 5.5 makes MiniLM find names it now misses, such as many of the 212 one-word given names it does not tag, the paper's pattern fails here, because that would mean an edit added knowledge MiniLM lacked. If Opus 5.5 targets the NRICs and cannot move them, the NRIC failure is missing knowledge rather than an obstacle. M3 records that MiniLM's training data held only 61 NRIC-shaped numbers inside the length limit, nearly all after the word "national", and M5 found that MiniLM never tags a Singapore NRIC as `national_id`, so that outcome is plausible too.

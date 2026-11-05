# M5 prediction for the 50-pass run

Written on 2026-11-05, before the M6 run, as §4 of the [brief](weight-space-pii-handoff.md) requires. M7 compares the edited MiniLM against this prediction. The base numbers come from `runs/base/baseline.json`, which `scripts/baseline.py` wrote from the locked sets.

The brief asks for this file in the run folder. `runs/` is gitignored, and the M4 driver refuses to start a run whose folder already exists, so the committed copy lives here. M6 copies it into `runs/p50/` after the run.

Revised on 2026-11-05, still before the M6 run, after a review of M5. The first version had two flaws. It called the regression check unchanged only if the edited F1 stayed within two paired standard errors. On 18,515 rows that error is so small that losing 10 entities out of 146,951 already counted as a change. Its range for overall Singapore F1 also did not follow from its own claims. The categories and their predictions are unchanged.

## What "fixed" means

A category counts as **fixed** if the share of its entities that MiniLM gets right rises by at least 10 percentage points on the main eval. It counts as **unchanged** if the share moves by less than that. A fall of 10 points or more counts as **broken**.

For the Nemotron scores, **unchanged** means the edited F1 is no more than 0.005 below or above the base F1. The score counts as **worse** only if it falls by more than 0.005 and by more than two paired bootstrap standard errors, so neither noise nor a negligible shift can trigger it. **Better** is the same rule upward. M7 reports the paired difference and its standard error either way.

## Prediction

Each share is the share of entities right on the main eval, 5,004 rows. Every prediction is a guess made before the run.

| Category | Entities | Base share right | Prediction | Reason |
|---|---|---|---|---|
| NRIC without tax context | 323 | 0% | Fixed | MiniLM finds the span but picks the wrong ID type: 168 of the 323 come out as `biometric_identifier`. That is an obstacle hiding an ability, like the paper's looping case. The 118 NRICs MiniLM misses completely stay missed. |
| NRIC with tax context | 711 | 87% | Unchanged | MiniLM already reads these as `tax_id`, which scores as right. An edit that pushes NRIC shapes toward `national_id` is still scored as right. |
| Given name of several words, later word | 1,434 | 5% | Unchanged | MiniLM types 1,227 of them as `last_name`. To fix this it would need to know which words are given names. Nemotron, its training data, almost never shows a given name of more than one word. That is missing knowledge, which the paper found editing cannot add. |
| Given name of one word | 3,376 | 62% | Unchanged | Most errors are unfamiliar names typed as `last_name` (766) or missed (212). Same missing knowledge. |
| Given name of several words, first word | 1,434 | 67% | Unchanged | Same as above. |
| Surname of one word | 2,688 | 89% | Unchanged | Already mostly right. |
| Surname of several words, first word | 1,274 | 81% | Unchanged | Errors split between `first_name`, `company_name` boundaries and misses. No single mechanism to target. |
| Surname of several words, later word | 1,734 | 83% | Unchanged | Same as above. |

## Overall scores

| Score | Base | Prediction |
|---|---|---|
| Singapore F1, main eval | 0.674 ± 0.004 | Rises by 0.002 to 0.011. Each NRIC without tax context that becomes right adds about 0.00005. "Fixed" needs at least 33 of the 323 right, which adds 0.0017. The 118 that MiniLM misses completely stay missed, so at most 205 can be fixed, which adds 0.0105. The range assumes every other kind stays where it is. |
| Nemotron F1, three scored labels | 0.984 ± 0.001 | Unchanged |
| Nemotron F1, all labels (regression check) | 0.952 ± 0.001 | Unchanged. The labels most at risk are the ones an NRIC edit could shift: `national_id` 0.960, `tax_id` 0.967, `biometric_identifier` 0.996, `health_plan_beneficiary_number` 0.988. |

## What would change my mind

If Opus 5.5 fixes the later words of given names, the paper's pattern fails here, because that would mean an edit added knowledge MiniLM lacked. If Opus 5.5 targets the NRICs and cannot move them, the NRIC failure is missing knowledge rather than an obstacle. M3 records that the training data held only 61 NRIC shapes inside the length limit, nearly all after the word "national", so that outcome is plausible too.

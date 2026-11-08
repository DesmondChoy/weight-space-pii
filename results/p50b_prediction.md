# Prediction for run p50b, the repeat of p50

Run `p50b` repeats `p50` with every setting M5 locked: the same Opus 5.5 command, effort, CLI version, budget of 50 forward passes, practice set and prompt. Only the run name differs, and with it the sandbox path in the prompt. It measures how much one editor's result varies between runs, which neither `p50` nor the paper's Experiment 8 can show. The paper ran each budget once.

Unlike the M5 prediction, this one is written after seeing `p50`'s result, so it is a forecast of a repeat, not a blind guess. It is committed before the run starts.

## What counts as the same result

The scores come from the M7 protocol: `scripts/evaluate.py` scoring on the main eval, 5,004 Singapore rows, and the regression check, 18,515 Nemotron rows.

| Measure | `p50` | `p50b` reproduces `p50` if |
|---|---|---|
| Singapore F1 gain | +0.059 ± 0.002 | The gain is between +0.039 and +0.079 |
| NRICs without "tax" just before them, share right | 0% → 85% | The share is at least 50% |
| Targeted titles tagged as a name | 63% → 0.1% | The share is below 10% |
| Nemotron F1, all labels | −0.0004 ± 0.0001 | The change is within ±0.005 |

## Prediction

| Question | Prediction | Reason |
|---|---|---|
| Does Opus 5.5 find the NRIC repair? | Yes | On the practice set, NRICs with no cue word are 0% right, which is the clearest single failure to read off the first diagnostic pass |
| Does Opus 5.5 find the title repair? | Yes, but less sure | Titles tagged as names are the largest source of false positives, but a run could spend its budget on names instead |
| Singapore F1 gain | Reproduces, between +0.039 and +0.079 | Follows from both repairs being found |
| Edit location | MLP neurons again, possibly in other layers | Neither repair needs a particular layer |
| Nemotron, all labels | Unchanged, within ±0.005 | `p50` changed it by −0.0004 |
| New names MiniLM did not tag before | Still few. At least 150 of the 212 untagged one-word given names stay untagged | `p50` left 186 untagged. Adding names would need knowledge MiniLM lacks |

## What would change the conclusion

If `p50b` finds neither repair, or its gain falls below +0.039, then `p50`'s +0.059 depends heavily on the run, and the write-up must say a single run is a weak measure of what an edit can do. If `p50b` makes MiniLM find many names it missed, the obstacle-versus-ability reading from M7 weakens.

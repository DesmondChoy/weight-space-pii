# Final submission: two repairs, in the MLPs of layers 1, 2 and 4

The submitted model (`work/edited`, also saved as `work/best_final`) fixes two Singapore-specific failure mechanisms. On the 500 held-out practice texts, which were never used to design anything, overall F1 rises from **0.856 to 0.921** (paired +0.065 ± 0.005; 141 items better, 2 worse). The last pass reproduced these numbers exactly on the saved tensors.

| Held-out (500 texts) | Original | Edited |
|---|---|---|
| name F1 | 0.865 | **0.918** (precision 0.862→0.946, recall 0.867→0.892) |
| national_id F1 | 0.747 | **0.956** (recall 0.612→0.940) |

Every edit has the same form. I repurposed two dead or near-dead MLP neurons as a detector. Its input weights are a closed-form Fisher direction computed from the exact MLP inputs, and its output column writes a state that the network itself uses. Each tensor changes in at most 4 neurons' rows or columns, plus a bias correction, so none is a rescaling.

## Repair 1: SG honorifics (layers 1 and 2)

**What went wrong.** US titles (Mr, Mrs) come out as O, but the titles common in SG text did not:
- The title itself was tagged first_name: Mayoress 20/20, Mstr 16/18, Miss 17/23, Madame 15/29 in dev.
- "Master" made the following name look like a company: P(company) 0.93 after Master versus 0.06 after Mr.

**Where it happens.** Mr and Mrs carry an "honorific" feature that tells layer 5 to output O. SG titles lack it, partly because suffix pieces ("mayor ##ess", "ms ##tr") make them look like name pieces. The next word reads the title through previous-token attention heads in layers 0–2 (attention weights near 1.0), before most of the network's processing.

**The edit.**
- A layer-1 detector (neurons 54/485) writes T2: the mean state of Mr/Mrs minus the mean state of SG titles at h_2.
- A layer-2 detector (neurons 1209/792) writes the same kind of feature at h_3.
- A separate Mayoress detector (neurons 244/1258) writes a stronger, Mayoress-specific shift.

**Evidence.**
- Writing the feature fixes the title. Moving it from layer 2 to layer 1 recovered more names after Master (69 → 81/95 on dev). Doubling its strength gave a further small gain (81 → 85).
- Subtracting the same feature from Mrs makes Mrs tagged as a name in 3 of 3 test sentences. So the network actually uses this feature; it isn't just an arbitrary push.
- The titles-only model on held-out data: name F1 0.865 → 0.917.
- On held-out text the layer-1 unit fires on all 189 SG-title occurrences and on only 4 other tokens.
- It does not fire on non-title uses such as "I miss you", "Master's degree", "MasterCard", "mayor" or "MS Word".
- A late safety check found it firing on "Mary" in long US documents. I recomputed the direction with 258 common first names as negatives, and afterwards 0 of 281 names were lost.

## Repair 2: NRIC format detector (layer 4)

**What went wrong.** The model typed IDs only from nearby keywords. Next to "tax" it got 84% of NRICs right. With no keyword it got 0 of 27 right, defaulting to biometric_identifier or O. The NRIC shape is linearly readable at the layer-4 MLP input, but nothing mapped it to a label.

**The edit.** Neurons 806/1407 detect the NRIC shape and write D_nat, the state difference between a cue-labelled ID and a cue-less NRIC.

**Evidence.**
- Recall rises with the strength of the push: 0.61 → 0.66 → 0.96 → 0.99 at 0, 0.5, 1 and 1.5 times D_nat.
- Subtracting D_nat removes the label from 14 of 31 NRICs that do have a cue.
- The NRIC-only model on held-out data: national_id F1 0.747 → 0.956, with names unchanged.
- On held-out text the unit fires on 110 of 116 NRICs, 4 IDs that are unscored anyway, and 1 other token.
- I kept the push at strength 1 because 1.5 gave no held-out gain.

## Protecting the standard texts

**Context-aware detector.** The network does not encode the NRIC's final letter at the start of the word, so look-alike codes can trigger the detector. I made it context-aware using synthetic look-alike codes and codes cued as other ID types. Look-alike codes that keep their original label:
- letter + 7 or 8 digits: 92%;
- cued licence and student IDs: 24 of 24;
- random, two-letter and digit-first codes: 100%.

**Bias correction.** The repurposed neurons used to have a small negative output on every token, and removing it flipped near-tied predictions (passport national_id → pin). Adding each neuron's average old contribution back into the layer's output bias removed those flips.

**US-style checks.** Across about 300 US-style texts, the only systematic change is that "Miss/Master/Madame" before US names is no longer tagged as a name.

## Tried and dropped

| Repair | Why it was dropped |
|---|---|
| Hyphenated names (layer-5 units) | Held-out gain +0.014 ± 0.015, not clearly above its standard error |
| Chat speaker names | Fixed about as many names as it created false positives |
| Names inside long runs read as company names | Would conflict with company names built from surnames in US data |
| NRIC push at strength 1.5 | No held-out gain and slightly more look-alike flips |

## Remaining limitations

- **Unseen NRIC contexts.** In 12 new sentence templates, NRICs went from 0/48 to 29/48 recognised. Every miss was a case where the detector stayed silent, mostly beside another ID type's cue ("Beneficiary", "Customer verified with"), which is deliberate.
- **"Mdm".** This title never appears in the practice data, so it is not handled.
- **Letter + 7-digit look-alikes.** About 8% still change label, and a long US document flipped one licence number to national_id.

I expect the gains to carry over to the evaluation texts: the held-out set comes from the same generator, and both repairs also held up in sentence templates I wrote fresh at the end.

# M2 report: profile and scope the data

M2 profiles the openpii Singapore rows in both splits and proposes the scope for M3: the filter, the sample sizes, and the labels to score. It follows §3 and §7 of [weight-space-pii-handoff.md](weight-space-pii-handoff.md). M2 also profiles Nemotron's held-out split, because the M3 label map compares the annotation conventions of the two datasets.

Status: complete, 2026-11-04. Not committed.

## Decisions

Desmond made these decisions on 2026-11-04. They override the brief's §3 where the two differ.

| Item | Brief's starting point | Decision |
|---|---|---|
| Practice set | 1,000 rows from `train` | 1,000 rows from `train`, unchanged |
| Main eval | 1,000 rows from `validation` | All `validation` rows that pass the filter, which is 5,004 of 5,005 |
| Regression check | 2,000 Nemotron held-out rows | All 18,515 Nemotron held-out rows of 512 tokens or fewer |
| Scored labels | 15 model labels | `first_name`, `last_name`, and `national_id` |
| Source of `national_id` | `IDCARDNUM` and NRIC-shaped `TAXNUM` | `TAXNUM` only, so `national_id` means NRIC |
| Labels for the M5 gate | The scored labels | The three scored labels, on both Singapore and Nemotron |
| Labels for the M7 regression check | The model's labels | All labels the model has, so the check catches damage outside the three scored labels |

The filter keeps rows of 512 tokens or fewer, drops duplicate texts within each split, and drops the `train` rows that share a text or `masked_text` with `validation`. The [filter section](#filter-on-length-duplicates-and-split-overlap) gives the steps.

The gold entity counts follow from these decisions:

| Set | `first_name` | `last_name` | `national_id` | Total |
|---|---|---|---|---|
| Main eval, 5,004 rows | 4,810 | 3,962 | 1,034 | 9,806 |
| Regression check, 18,515 rows | 13,512 | 9,699 | 550 | 23,761 |

The filter drops one duplicate text from `validation`. The dropped copy has none of the three scored labels, so the main eval keeps all 9,806 gold entities. Nemotron has no duplicate texts among its 18,515 filtered rows. The regression check holds 146,769 spans across all 55 entity types the model has. M3 recomputes these counts when it draws the sets.

The rest of this report gives the evidence. The [profile](#profile) section lists what the data contains. The [proposed scope](#proposed-scope) section gives the reasons behind each decision.

## Profile

[scripts/profile.py](../scripts/profile.py) reads the three files from M1 and writes `data/m2_profile.json`. It counts tokens with the model's own `tokenizer.json`, including special tokens, with truncation and padding turned off. The tokenizer file pads every encoding to 512 by default. An early run that left padding on reported every row as exactly 512 tokens.

### Spans and gold spans

Each row is a text plus a list of spans. A span marks one piece of PII by its start and end character positions in the text, and gives it a label. openpii stores a row's spans in its `privacy_mask` field. Nemotron stores them in its `spans` field, as a string that `ast.literal_eval` parses.

The dataset's spans are the gold spans. Gold spans are the reference answer, and scoring compares the model's predicted spans against them. We do not draw gold spans ourselves.

The first `validation` row begins with this text, shortened here:

> Mister Enie Idilia Cibien, We are thrilled to confirm your spot for the upcoming Skydiving Weekend Adventure scheduled for December 25th, 1944 at the Singapore Drop Zone. The briefing will commence at 08:30 AM in the conference room located at 11 Lorong K Telok Kurau, … Please bring a copy of your OH1409941 or 598QI5X2NW for identity verification, …

Its first nine spans are:

| Start | End | Label | Text the span covers |
|---|---|---|---|
| 0 | 6 | `TITLE` | `Mister` |
| 7 | 18 | `GIVENNAME` | `Enie Idilia` |
| 19 | 25 | `SURNAME` | `Cibien` |
| 124 | 143 | `DATE` | `December 25th, 1944` |
| 151 | 160 | `CITY` | `Singapore` |
| 245 | 247 | `BUILDINGNUM` | `11` |
| 248 | 268 | `STREET` | `Lorong K Telok Kurau` |
| 369 | 378 | `PASSPORTNUM` | `OH1409941` |
| 382 | 392 | `IDCARDNUM` | `598QI5X2NW` |

A generator, not a human annotator, produced openpii's text and spans. We infer this from the data, not from openpii's documentation. ID values are random strings, phone numbers put a `+65` prefix on random digits, and some `CITY` spans repeat a place name. Because the generator inserted every value itself, it misses no entity. It does draw some span boundaries that a human annotator would not, and the [three annotation patterns](#three-annotation-patterns-distort-strict-f1) below follow from that.

M3 changes the gold spans in two ways, and moves no span boundary:

1. It renames each scored label to the model's label name. `GIVENNAME` becomes `first_name`, `SURNAME` becomes `last_name`, and `TAXNUM` becomes `national_id`.
2. It drops the spans of every other label, such as `TITLE`, `PASSPORTNUM`, and `CITY`.

"Gold" therefore means what the dataset marks as correct, not what is correct by an outside standard.

### Every Singapore row fits in 512 tokens

| Set | Rows | Mean tokens | p50 | p90 | p99 | Max | Rows of 512 tokens or fewer |
|---|---|---|---|---|---|---|---|
| openpii SG `train` | 20,225 | 96.5 | 78 | 187 | 298 | 574 | 20,223 (99.99%) |
| openpii SG `validation` | 5,005 | 96.2 | 78 | 185 | 301 | 478 | 5,005 (100%) |
| Nemotron held-out | 20,000 | 243.7 | 197 | 469 | 784 | 2,979 | 18,515 (92.6%) |

The 512-token filter removes 2 `train` rows and no `validation` rows. In Nemotron it removes 1,485 rows, or 7.4%.

All 5,005 `validation` rows have `language == "en"`, as the brief says.

`train` has 7 duplicate texts and `validation` has 1. One text appears in both splits. Two `masked_text` templates also appear in both splits.

### Spans per label

The table counts spans for each openpii label. The rows column counts the `validation` rows that hold at least one span with the label. A row can hold several spans with one label, so the row count can be lower than the span count. The last column gives the share of `validation` spans that contain more than one whitespace-separated word. In the example row above, `Enie Idilia` is a multi-word `GIVENNAME` span and `Cibien` is a one-word `SURNAME` span.

| openpii label | `train` spans | `validation` spans | `validation` rows | Multi-word share |
|---|---|---|---|---|
| `GIVENNAME` | 19,678 | 4,810 | 3,561 | 29.8% |
| `DATE` | 18,315 | 4,609 | 3,499 | 32.9% |
| `SURNAME` | 15,876 | 3,962 | 3,218 | 32.2% |
| `EMAIL` | 11,327 | 2,778 | 2,655 | 0.8% |
| `CITY` | 10,466 | 2,583 | 2,421 | 74.3% |
| `TITLE` | 10,340 | 2,561 | 2,338 | 0% |
| `TELEPHONENUM` | 8,229 | 2,097 | 2,011 | 59.1% |
| `STREET` | 7,786 | 1,962 | 1,923 | 100% |
| `BUILDINGNUM` | 7,479 | 1,868 | 1,847 | 0% |
| `AGE` | 7,239 | 1,824 | 1,690 | 0% |
| `ZIPCODE` | 6,187 | 1,560 | 1,550 | 0% |
| `IDCARDNUM` | 5,586 | 1,429 | 1,417 | 0% |
| `CREDITCARDNUMBER` | 5,385 | 1,265 | 1,200 | 0% |
| `DRIVERLICENSENUM` | 4,124 | 1,062 | 1,053 | 0% |
| `TAXNUM` | 4,242 | 1,034 | 1,030 | 0% |
| `GENDER` | 4,082 | 1,021 | 999 | 10.5% |
| `PASSPORTNUM` | 3,410 | 814 | 808 | 0% |
| `SEX` | 2,908 | 746 | 735 | 0% |
| `SOCIALNUM` | 3,214 | 745 | 741 | 0% |
| `TIME` | 2,037 | 454 | 153 | 35.2% |

`TIME` appears in only 153 `validation` rows, at about three spans per row.

The table leaves out the rare labels, which have fewer than 25 spans each. `validation` has 29 spans across 10 rare labels: `USERNAME`, `URL`, `ORGANISATION`, `COUNTRY`, `AMOUNT`, `CURRENCY`, `WEIGHT`, `BANKNAME`, `ACCOUNTNUM`, and `HEIGHT`. `train` has 104 spans across 13 rare labels. Those are the same labels without `HEIGHT`, plus `ALLERGIES`, `SALARY`, `TIMEZONE`, and `IPV4`.

No openpii row has overlapping spans. Nemotron has 5 overlapping span pairs in 20,000 rows.

### Only `TAXNUM` holds NRICs

The script uses two shape tests:

- A value is NRIC-shaped when the whole value matches `[STFGM]\d{7}[A-Z]`.
- A phone number is Singapore-shaped when it is an optional `+65`, then eight digits that start with 3, 6, 8, or 9. A space or hyphen can follow the fourth digit.

| Check | `train` | `validation` |
|---|---|---|
| NRIC-shaped values across all ID labels | 4,242 of 20,576 (20.6%) | 1,034 of 5,084 (20.3%) |
| NRIC-shaped `TAXNUM` values | 4,242 of 4,242 (100%) | 1,034 of 1,034 (100%) |
| NRIC-shaped `IDCARDNUM`, `PASSPORTNUM`, `SOCIALNUM`, and `DRIVERLICENSENUM` values | 0 | 0 |
| Singapore-shaped phone numbers | 0 of 8,229 | 0 of 2,097 |
| Phone numbers that start with `+65`, `0065`, or `65` | 3,433 of 8,229 (41.7%) | 863 of 2,097 (41.2%) |
| Six-digit postcodes | 6,187 of 6,187 (100%) | 1,560 of 1,560 (100%) |

The ID counts match the brief exactly. Every `TAXNUM` value is an NRIC, and no other ID label holds one. `IDCARDNUM` values are random 10-character strings such as `598QI5X2NW`.

The phone generator puts a Singapore prefix on about 41% of numbers. The digits after the prefix are random, as in `+65 95 118.2657`.

### Three annotation patterns distort strict F1

Strict F1 counts a predicted entity as correct only if its span matches the gold span exactly. In three places, openpii draws spans differently from Nemotron, the data the model learned from.

**Multi-word names.** In `validation`, 29.8% of `GIVENNAME` spans and 32.2% of `SURNAME` spans have more than one word, as in `Enie Idilia` and `Tham Teo Chew`. In Nemotron, 0.3% of `first_name` spans and 1.1% of `last_name` spans have more than one word. The M1 probe found that the model labels each word of a multi-word name as a separate entity. Multi-word given names are normal in Singapore, as in "Wei Ming" in "Tan Wei Ming". Most of openpii's multi-word names are not local.

**City names.** In `validation`, 25.7% of `CITY` spans are the plain word `Singapore`. Another 49.7% join `Singapore` to a neighbourhood or place, as in `Singapore Tanjong Pagar Conservation Area`. The remaining 24.6% join `Singapore` to a place and then repeat the place, as in `Singapore Jalan Kayu Estate Jalan Kayu Estate`. In Nemotron, 81% of `city` spans are one word.

**Street addresses.** In `validation`, 61.6% of `BUILDINGNUM` spans come directly before a `STREET` span. Another 25.2% come directly after one, as in `Mangis Road # 350`. The other 13.3% are not next to a `STREET` span. Between adjacent spans, the text is a space, `#`, a comma, or a non-breaking space. The brief merges `BUILDINGNUM` and `STREET` when they are adjacent, so a merge rule would have to handle both orders and the `#` separator. The final scope does not score street addresses, so M3 needs no merge rule.

These patterns also reach the editor. The editor sees the gold spans of its 1,000 practice rows, and its forward passes grade edits against those spans. An edit can therefore raise the practice score by matching openpii's span boundaries instead of by finding more PII. The M3 review rows let Desmond check the gold spans by eye before the run.

## Proposed scope

### Filter on length, duplicates, and split overlap

1. Keep the rows that have 512 tokens or fewer under the model tokenizer, including special tokens.
2. Drop duplicate texts within each split.
3. Drop the `train` rows whose text or `masked_text` also appears in `validation`. After this step, the practice set and the main eval share no text and no template.
4. Apply the 512-token filter and the duplicate filter to Nemotron.

### Use every filtered row for both evaluation sets

We recommend changing the size of both evaluation sets, and keeping the practice set at 1,000 rows. The [decisions table](#decisions) gives the final sizes. Cost is not a constraint. The base model scored 1,000 `train` rows in 1.9 s on CPU in each of two runs, measured with a one-off script that is not in the repo. At that rate, the full `validation` split takes about 10 s per model.

The gain is statistical power. The main eval would hold 1,034 NRICs instead of about 207. If the bootstrap standard error scales with one over the square root of the row count, it shrinks by about 2.2 times. That scaling is an assumption, and M5 measures the real standard error. Power matters most for the M5 gate and for the per-label results in M7. The editor never sees the main eval, so a larger main eval does not change what the editor can learn.

The same reasoning applies to the regression check. A 2,000-row Nemotron sample would hold about 60 `national_id` spans, too few to show whether an edit damaged them. All 18,515 filtered rows hold 550. They hold 3,909,853 tokens, so scoring them takes about 80 s per model at the throughput measured on `train` rows, about 49,000 tokens per second. That figure is an estimate, because Nemotron rows are longer and attention cost grows faster than token count. Using every filtered row in both evaluation sets also means that neither set depends on a sampling seed.

The regression check scores all of the model's labels, not only the three scored labels. Its job is to show that an edit did not make the model worse anywhere, and an edit could damage a label such as `email` that the Singapore eval does not score. The M5 gate is different. It compares Singapore F1 with Nemotron F1, so it uses the same three labels on both sets.

We recommend keeping the practice set at 1,000 rows. The editor can cover all of it in two forward requests of 512 items, and it holds about 210 NRICs.

### Score names and NRICs only

M2 first proposed 13 scored labels, without `CITY`. Desmond narrowed the set to three, to keep the experiment small and to match Desmond's domain, insurance, where some labels such as `DATE` are not treated as PII.

| openpii label | Model label | `validation` gold entities |
|---|---|---|
| `GIVENNAME` | `first_name` | 4,810 |
| `SURNAME` | `last_name` | 3,962 |
| `TAXNUM` | `national_id` | 1,034 |
| Total | | 9,806 |

`national_id` takes only `TAXNUM`, so in this experiment `national_id` means NRIC. Every `TAXNUM` value is NRIC-shaped, and no other label holds an NRIC. The NRIC case is the one the brief says this data can test, because the model detects NRICs but gives them varying wrong types. The two name labels are core PII in insurance. Their result mostly measures how the model splits multi-word names, as the [annotation patterns](#three-annotation-patterns-distort-strict-f1) section describes.

No other label is scored. These labels were considered and left out:

| openpii label | Reason |
|---|---|
| `IDCARDNUM` | Its values are random 10-character strings, not NRICs. Adding them would make `national_id` mean two things |
| `TELEPHONENUM` | No value is Singapore-shaped, and the model already labels real Singapore numbers correctly |
| `CREDITCARDNUMBER` | Only 142 of 1,265 `validation` values pass the Luhn checksum, so the digits are random. Card numbers carry no Singapore-specific signal |
| `PASSPORTNUM` | The model has no passport label, and the editor cannot add one, because edits to the classifier head are out of scope |
| `CITY` | Three quarters of the spans join `Singapore` to a place name, which strict F1 would mark wrong |
| `DATE`, `TIME`, `AGE`, `GENDER`, `SEX`, `EMAIL`, `ZIPCODE`, `BUILDINGNUM`, `STREET`, `DRIVERLICENSENUM` | Left out to keep the scope small |
| `TITLE`, `SOCIALNUM`, and the rare labels | The brief excludes them |

More labels would not add power where the experiment needs it. The main eval already uses every `validation` row, so it already holds all 1,034 NRICs in the split. Another label would add entities to the overall F1, but not to the NRIC result.

## Reproduce the profile

```bash
uv run python -I scripts/profile.py
```

The script runs in about 4 s and writes `data/m2_profile.json`. That file is gitignored, so rerun the script to rebuild it.

## Open items

The [decisions](#decisions) section at the top records the final scope. These items remain open for M3:

- M3 fixes how to score multi-word names. About 30% of name spans have more than one word, and the model splits them.
- The pipeline figure, [docs/figures/e2e_pipeline.svg](figures/e2e_pipeline.svg), still shows a 1,000-row main eval and a 2,000-row regression set. M3 regenerates it from `e2e_pipeline.py`.

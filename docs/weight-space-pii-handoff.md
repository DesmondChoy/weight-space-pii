# Handoff: Weight-Space Programming on a PII detector (MVP)

You're helping me (Desmond) run a minimal version of MIT CSAIL's **"Sparks of Weight-Space Programming"**, Experiment 8, on a different model and task.

- Blog post: https://rikiyat.github.io/blog/weight-space-programming/
- Full logs: https://github.com/RikiyaT/lm_understand-logs. The closest template is `exp8_math/`. Read its `README.md` and `runs/p50c/turns/t001/prompt.txt` before you start.

The harness copies Experiment 8's rules but not its release tooling. The dataset is openpii's Singapore rows, used as they are. Don't add anything beyond this brief unless I ask. The longer first draft is in `weight-space-pii-handoff.full.md`, for reference only.

---

## 1. What we're testing

In Experiment 8, a frontier LLM had the weights of Qwen3.5-0.8B in a Python kernel, practice problems, and a budget of forward passes. Its job was to make the model better at math **by editing weights directly, with no training**. The paper found that editing could remove an obstacle that hid an existing ability (looping), but could not add an ability the model lacked.

Our version:
- The model we edit is `Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection`. It's a token-classification PII detector fine-tuned on `nvidia/Nemotron-PII`, which uses US personas. It has 6 layers, width 384, 12 heads, MLP width 1,536, and 111 BIO labels covering 55 entity types.
- The editor's goal is better PII detection in **Singapore-context English text**.
- The editor is Claude Opus 5.5 (`claude-opus-5-5`), not Astra.

The question is whether the paper's obstacle-versus-ability pattern holds here. A quick probe suggests it might split by category. NRICs get detected but fragmented across several ID types, which looks like an obstacle. Local naming conventions like `s/o` and `binte` get mislabelled, which may be missing knowledge. openpii's Singapore rows contain few local names, so the NRIC case is the one this data can test.

---

## 2. Harness

You (this session) are the **builder**. You build the harness and run it. The **editor** is Opus 5.5, run through the Claude Code CLI with all of its tools switched off. This mirrors the paper, which used `codex exec` with Codex's own tools off. The editor never sees this brief, the eval sets, or the scores.

**Editor command.** The driver starts each conversation with a fresh session ID, and sends each later message with `--resume` on that ID:

```bash
claude -p "<message>" --session-id <uuid> --model claude-opus-5-5 --effort max --tools "" --strict-mcp-config --disable-slash-commands --output-format json
```

- Run it from an empty directory outside this project, so the editor loads no `CLAUDE.md`, project settings, or memory.
- Don't use `--bare`, because it requires an API key. The CLI signs in with the claude.ai subscription.
- `--output-format json` returns the reply text, token usage, and session ID for `run.json`.
- If a call fails or hits a usage limit, wait and resume the same session. Don't count the failed call as a turn.

**Driver loop.**
1. Send the task prompt from §6.
2. The editor replies with exactly one ` ```python ` block, one ` ```forward ` block, or, at the end, a final answer with no block.
3. Run the block. Send back the printed output or the forward-pass summary as the next user message.
4. Repeat until the budget is spent, then collect the final answer.
5. If the context fills, start a fresh conversation with the task, `/work/notes.md`, and the current edit list. Build this only if a run actually hits the limit. The CLI may compact a long conversation on its own instead. If it does, record that in `run.json`.

**Kernel.** A persistent Python subprocess (variables survive between blocks), CPU only. Run it with gradients disabled (`torch.set_grad_enabled(False)`). Model files are read-only in `/model`, practice data is read-only in `/practice`, and `/work` is read-write. Docker isn't needed.
- Predefined names are `NAMES`, `load(name)`, `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()`, and `tok`.
- Limits are 300 code runs, 900 seconds per run, and 10,000 characters of output.

**`save_edit` refuses** the same kinds of edit the paper put out of scope:
- the embeddings (`*embeddings*`),
- the classification head (`classifier.*`),
- any edit that only multiplies a whole tensor by a constant.

**Forward-pass server.** This is the only way the model runs. It has one request type, `predict`, specified in §6. Each request costs one pass. A failed request costs nothing.

**Logging.** Write one folder per run:
- `transcript.jsonl` has every message, code block, output, forward request, and forward summary, in order.
- `forward/` has each request and its full result.
- `edit/` has `manifest.json` (changed tensors, entries changed, hashes) and `edit.npz`.
- `final_answer.md` is the editor's last reply.
- `run.json` records the model, effort, tokens, and wall time.

---

## 3. Data: openpii's Singapore rows, used as they are

**Source.** `ai4privacy/pii-masking-openpii-1.5m` (CC-BY-4.0; attribute "Ai4Privacy / Ai Suisse SA"), rows with `region == "SG"`. As in the paper, the data is used as it is, and the editor finds the failures itself. All 5,005 SG rows in the `validation` split are English. What I measured in that split:
- Streets and postcodes are local, for example "Lorong K Telok Kurau" and "641221".
- 1,034 of the 5,084 ID values are NRIC-shaped, and all of them are labelled `TAXNUM`.
- Names are mostly not local, for example "Enie Idilia Cibien".
- None of the 2,097 phone numbers are Singapore-shaped.

**Sample to a manageable size.** Use a fixed-seed sample. The starting scope is below, and M2 adjusts it from the data profile.
- Practice is 1,000 rows from the `train` split.
- Eval is 1,000 rows from the `validation` split.
- Keep only rows that fit in 512 tokens.

Record the seed and the row count after each filter.

**Relabel into the model's labels.** Map openpii labels onto the model's label names, and don't score labels with no match. This is the starting map. Fix the full map at M3, including how to treat confusable pairs such as `date` and `date_of_birth`.

| openpii label | Model label |
|---|---|
| `GIVENNAME` | `first_name` |
| `SURNAME` | `last_name` |
| `TELEPHONENUM` | `phone_number` |
| `IDCARDNUM`, NRIC-shaped `TAXNUM` | `national_id` |
| other `TAXNUM` | `tax_id` |
| `BUILDINGNUM` and `STREET`, merged when adjacent | `street_address` |
| `CITY` | `city` |
| `ZIPCODE` | `postcode` |
| `EMAIL` | `email` |
| `DATE` | `date` |
| `TIME` | `time` |
| `AGE` | `age` |
| `GENDER`, `SEX` | `gender` |
| `CREDITCARDNUMBER` | `credit_debit_card` |
| `DRIVERLICENSENUM` | `certificate_license_number` |
| `TITLE`, `PASSPORTNUM`, `SOCIALNUM`, rare labels | not scored |

**Splits.**

| Role | Contents | Seen by editor? |
|---|---|---|
| Practice | 1,000 sampled rows from openpii `train`, SG | Yes, in `/practice` |
| Main eval | 1,000 sampled rows from openpii `validation`, SG | Never |
| Regression check | Fixed sample of 2,000 from the model's own 20k held-out split of `nvidia/Nemotron-PII`. The model's training notebook merged `train` and `test` and re-split 90/10 with seed 42, so 89,895 of the 100,000 `test` rows are training data. | Never |

Before any run, confirm that Nemotron's span labels match the model's label names. Score Nemotron only over labels the model has.

---

## 4. Runs

- **Base model.** Evaluate it once on all sets.
- **One edited run at 50 forward passes.** In the paper, only the 50-pass run moved the results much. Add a 10-pass run only if the 50-pass run shows a clear gain and I ask for the budget curve.

Before the edited run, write down a prediction for each category: will the edit fix it, or will it stay unchanged? Commit the prediction to the run folder. The write-up compares the result against it.

---

## 5. Evaluation and reporting

**Fixed protocol**, the same for every model:
- the model's tokenizer, max length 512, argmax label per token, standard BIO repair, and word-level spans,
- entity-level strict micro **F1** (exact span and type), plus precision and recall.

**Main table.**

| Model | Singapore F1 | Nemotron F1 | Mechanism metric | Weights changed |
|---|---|---|---|---|
| Base | | | | none |
| Edited, 50 passes | | | | |

- Choose the **mechanism metric** after reading the final answer, as the paper did with "hit token limit %". It is the share of whatever failure the editor targeted, for example the share of NRICs predicted as one `national_id` span.
- The **appendix** has per-category F1 for both sets, what the run changed (from `final_answer.md`), and the prediction from §4 next to the result.

---

## 6. The editor's task prompt

Adapted from the paper's Experiment 8 prompt (`exp8_math/runs/p50c/turns/t001/prompt.txt`, CC BY 4.0). Fill in every `{…}` from the real model config and data, then show it to me before the run.

```
TASK. You are given the weights of a PII detector: a token-classification model fine-tuned from all-MiniLM-L6-v2 ({n_layers} BERT layers of width {hidden}, {n_heads} attention heads, MLP width {intermediate}, vocabulary {vocab}; {n_labels} BIO labels: {label_list}). It was trained on English text built from United States personas. Your job is to make it better at detecting personal information in English text from Singapore by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it fails on Singapore text and repair that with targeted changes.

HOW IT WILL BE MEASURED. After you finish, your edited model and the unedited model are evaluated on held-out English texts from Singapore with annotated personal information, and on standard PII texts to check that nothing else got worse. The metric is entity-level F1: a predicted entity counts only if its span and its type both match the annotation exactly. It is computed over these categories: {scored_categories}. The tokenizer, maximum length ({max_len}), decoding (argmax label per token, BIO repair as follows: {bio_rule}) and the label set are fixed: only weight values can change (same tensor names and shapes). You never see these evaluation sets.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it fails on Singapore text, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Repairing a specific failure mechanism is in scope: for example, if the model consistently splits or mistypes a kind of local identifier or name, finding and fixing what causes that counts. Out of scope, and refused by save_edit: the embeddings ({embedding_tensor_names}), the classification head ({classifier_tensor_names}), and any edit that only multiplies a whole tensor by a constant. Global knobs (making the model flag more or fewer tokens everywhere, regardless of context) are not what we are looking for.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. The model's files are read-only in /model, practice data read-only in /practice, and /work is your read-write folder. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, safetensors, tokenizers. Defined in advance: NAMES (all tensor names), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() and tok (the tokenizer). Gradients are disabled. To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an expression, its value is shown. One execution may take up to 900 seconds; output is cut at 10,000 characters. You may run code up to 300 times. If the kernel is restarted, its variables are lost and you are told so.
2. Practice data with annotations (none of it is in the evaluation), one JSON object per line with id, text and spans (start, end, label), using the model's label names: /practice/sg_train.jsonl ({n_practice} English texts from Singapore). Choose whichever serve your experiments.
3. A forward-pass tool with a budget of {budget} forward passes in total. Each request counts as one pass. Reply with exactly one block that starts with ```forward and contains JSON:
 {"model": "original", "edited" or "both", "items": ["<practice id>", ...] (optional), "texts": ["<your own text>", ...] (optional, ungraded), "return": {"label_probs": true or false, "hidden_states": [layer indices 0..{n_layers}], "mlp_act": [layers], "attention": [layers; at most 64 texts]} (optional)}
 Up to 512 items and texts in total. For graded items you get entity-level precision, recall and F1 (overall and per category) and failure counts (missed entity, wrong type, wrong boundary, false positive); per-item predictions and grades are in /work/pred_<k>.jsonl. "both" runs the unedited and the edited weights on the same items and also returns the F1 difference (edited minus original) with its bootstrap standard error over items, and how many items got better or worse. Requested arrays are saved as .npy files with meta.json in /work/fwd_<k>/ (0 = embeddings, l = after layer l). A request that fails does not count against the budget.

HOW TO WORK.
1. Diagnose first. Look at how the model fails on practice texts: missed entities, wrong types, wrong boundaries, false positives, and which categories and local formats are involved. Read some failing examples. Go after the largest failure you can explain.
2. Localize causally. That a component is more active on correct predictions is not enough. Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. Compare original and edited weights on the SAME items: a request with "model": "both" does this and reports the paired difference with its standard error. Keep an edit only if its gain is clearly larger than its standard error on items that were not used to choose it. Check that an edit does not simply flag more tokens: judge edits by F1, and watch false positives. Edits that change a tensor by a tiny fraction usually have effects too small to see; test strengths large enough to matter and back off if they hurt.
4. Plan your forward passes: for example one diagnostic pass with activations, passes on your own texts for dense measurements, and a final "both" pass on items you have not used, to confirm.
5. Work efficiently. Before each code block, explain the step in plain English; state the hypothesis of each code block in its first comment line; load things once into kernel variables; print compact summaries, never whole arrays, and short text excerpts only (at most about 300 characters each). Keep /work/notes.md up to date (findings with numbers, edits tried and their measured effect, the current best edit); it is shown to you again if the conversation is restarted. Keep copies of good edits (for example copy /work/edited to /work/best_1/).

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) internet access or search (there is none). Allowed: any analysis of the weights and of forward-pass results, any use of the practice data, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. It continues until all {budget} forward passes are used (replying without a code block does not end it earlier). After that, make your final choice (the tensors in /work/edited when you finish are submitted) and reply without a code block, saying what you changed, the evidence for it and why you expect it to help.
```

---

## 7. Milestones

Work in eight milestones. At the end of each one, **⏸ stop**, show me what the milestone lists under "Show me", and wait for my go-ahead. Don't start the next milestone early, and don't change anything an earlier milestone froze without asking me.

**M1. Environment.**
- Create `.venv` with `uv`, and pin every dependency in `pyproject.toml` and `uv.lock`.
- Download the model at a pinned revision, the openpii SG rows from both splits, and the model's Nemotron-PII held-out split, rebuilt from its training notebook.
- Check that `claude auth status` reports `"loggedIn": true`. The CLI isn't signed in on this Mac yet.
- Show me the lockfile, the model and dataset revision hashes, and the base model's labels for three hand-written Singapore sentences with a name, an NRIC, and a phone number.

**M2. Profile and scope the data.**
- Profile the SG rows in each split: row counts, length in tokens, label counts, and the share of IDs that are NRIC-shaped.
- Propose the scope: the filter, the sample sizes, and the labels to score. §3 gives the starting point.
- Show me the profile and the proposed scope.

**M3. Sample, relabel, define the metric, and review samples.**
- Draw the samples with a fixed seed and relabel them, as §3 describes.
- Draw the Nemotron sample and run the label check from §3.
- Write the scoring code from §5. Test it on hand-built cases, including an exact match, a wrong type, a boundary off by one word, a missed entity, and a false positive.
- Ask me how many rows to review. Pick that many sampled rows, covering every scored label if the number allows. Render each row with its gold spans, and give the label you expect the base model to output for each entity.
- Show me the row count after each filter, the full label map, the Nemotron label check, the metric test results, and the review rows. I check the gold labels and the label map. If I change the map, redo the relabel and the checks.

**M4. Harness and dry run.**
- Build the driver, kernel, `save_edit` refusals, forward-pass server, and logging from §2.
- Check that each refusal fires, using one test edit per refused case.
- Check that the forward-pass server and the scoring code give the same predictions and scores on the same 50 practice items.
- Do a dry run with a budget of 2 passes and at most 10 turns, to test the loop end to end. Delete its edits afterwards.
- In the dry run, ask the editor which tools, instructions, and context it has. Confirm that no hooks, plugins, skills, or memory reach it.
- Show me the filled-in prompt, the refusal tests, the server and scoring check, and the dry-run transcript.

**M5. Freeze and baseline.**
- Write `lock.json` with hashes of the eval sets, the scoring code, the forward-pass server, and the prompt. Nothing in it changes after this point.
- Score the base model on all sets.
- Check that there is something to repair. On the scored labels, Singapore F1 must be lower than Nemotron F1 by more than twice its bootstrap standard error. If it isn't, stop, because the data needs rethinking.
- Write the per-category prediction from §4.
- Estimate the cost and wall time of the editor run from the dry run's tokens per turn.
- Ask me how many failing rows to show.
- Show me the baseline table, per-category scores, that many failing rows, the prediction, and the cost estimate.

**M6. Editor run.**
- Run the 50-pass session.
- Show me `final_answer.md` and the edit manifest.

**M7. Evaluate the edited model.**
- Rebuild the edited weights from `edit/edit.npz` and the original weights, and check that the hashes match the manifest.
- Score the edited model on the locked sets. Report the paired difference against the base model, with its bootstrap standard error.
- Choose the mechanism metric from the final answer, as §5 describes.
- Ask me how many changed rows to show.
- Show me the main table, the per-category results next to the prediction, and that many rows the edit changed, both fixed and broken.

**M8. Write-up and demo.**
- Write the write-up and build the demo from §8.
- Show me both.

---

## 8. Deliverables

1. The run folder in the layout from §2.
2. The main table from §5, plus the appendix.
3. A short write-up covering:
   - what the run diagnosed and changed,
   - whether it worked, category by category, against the prediction,
   - how it compares with the paper's Experiment 8, including whether we see the obstacle-versus-ability split,
   - limitations. openpii's Singapore rows have local streets, postcodes, and some NRICs, but mostly non-local names and no local phone numbers, so the result describes this dataset and not Singapore text in general, and there is one run with one editor.
4. **Demo.** An HTML page showing a fictional Singapore claims letter, highlighted by the base model and the edited model side by side.

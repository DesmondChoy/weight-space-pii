"""Run the token classifier and decode word-level entities under the fixed protocol."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from wsp.data import MAX_LEN, MODEL_DIR, Span
from wsp.score import decode


def load(model_dir=MODEL_DIR):
    torch.set_grad_enabled(False)
    return AutoTokenizer.from_pretrained(model_dir), AutoModelForTokenClassification.from_pretrained(model_dir).eval()


def word_tags(enc, i: int, argmax: list[int], id2label: dict) -> tuple[list[tuple[int, int]], list[str]]:
    """Each word's character span and the tag of its first subtoken."""
    words: dict[int, list[int]] = {}
    tags: dict[int, str] = {}
    for t, w in enumerate(enc.word_ids(i)):
        if w is None:
            continue
        a, b = enc["offset_mapping"][i][t].tolist()
        if w not in words:
            words[w] = [a, b]
            tags[w] = id2label[argmax[t]]
        words[w][1] = b
    order = sorted(words)
    return [tuple(words[w]) for w in order], [tags[w] for w in order]


def predict(tok, model, texts: list[str], batch: int = 32) -> list[list[Span]]:
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[list[Span]] = [[] for _ in texts]
    for k in range(0, len(order), batch):
        ids = order[k:k + batch]
        enc = tok([texts[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                  return_offsets_mapping=True, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       token_type_ids=enc.get("token_type_ids")).logits
        argmax = logits.argmax(-1).tolist()
        for j, i in enumerate(ids):
            words, tags = word_tags(enc, j, argmax[j], model.config.id2label)
            out[i] = decode(words, tags)
    return out

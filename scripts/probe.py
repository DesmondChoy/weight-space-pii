"""Print base-model labels for hand-written Singapore sentences.

Shows the argmax label of each word's first subtoken, and the entities formed by
B-/I- runs of one type. This is a sanity check, not the M3 scoring protocol.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "model"

SENTENCES = [
    "My name is Tan Wei Ming and my NRIC is S8812345D; call me at +65 9123 4567.",
    "Please update the policy for Siti Nurhaliza binte Ahmad, NRIC T0234567J, mobile 8765 4321.",
    "Claimant Rajesh s/o Kumaran (S7654321A) can be reached on 6123 4567 or +65 8234 5678.",
]


def word_labels(text, tok, model):
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        pred = model(**enc).logits[0].argmax(-1).tolist()
    words, seen = [], set()
    for i, w in enumerate(enc.word_ids()):
        if w is None or w in seen:
            continue
        seen.add(w)
        span = enc.word_to_chars(w)
        words.append((span.start, span.end, model.config.id2label[pred[i]]))
    return words


def entities(text, words):
    ents = []
    for start, end, label in words:
        tag, _, typ = label.partition("-")
        if tag == "I" and ents and ents[-1][2] == typ and ents[-1][3] == "open":
            ents[-1][1] = end
        elif tag in ("B", "I"):
            ents.append([start, end, typ, "open"])
        else:
            if ents:
                ents[-1][3] = "closed"
    return [(text[s:e], typ) for s, e, typ, _ in ents]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).eval()
    for text in SENTENCES + sys.argv[1:]:
        words = word_labels(text, tok, model)
        print(f"\n{text}")
        print("  words: " + " ".join(f"{text[s:e]}/{l}" for s, e, l in words))
        for span, typ in entities(text, words):
            print(f"  entity: {span!r} -> {typ}")


if __name__ == "__main__":
    main()

"""The editor's task prompt: the brief's §6 template, filled from the model config, the weights and the practice set."""

import json
import re
from pathlib import Path

from safetensors import safe_open

from wsp.data import MAX_LEN, read_jsonl
from wsp.score import SG_LABELS

TEMPLATE = Path(__file__).with_name("task_prompt.txt")
BIO_RULE = ("each word, as the tokenizer's pre-tokenizer splits text at whitespace and punctuation, takes the label "
            "of its first subword token; an I-X label continues an entity only directly after a word labelled B-X or "
            "I-X, and otherwise starts a new X entity")


def fields(budget: int, box) -> dict[str, str]:
    cfg = json.loads((box.model / "config.json").read_text())
    labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    types = [lab[2:] for lab in labels if lab.startswith("B-")]
    with safe_open(box.model / "model.safetensors", "pt") as f:
        names = list(f.keys())
    practice = box.practice / "sg_train.jsonl"
    return {
        "n_layers": str(cfg["num_hidden_layers"]),
        "last_layer": str(cfg["num_hidden_layers"] - 1),
        "hidden": str(cfg["hidden_size"]),
        "n_heads": str(cfg["num_attention_heads"]),
        "intermediate": f"{cfg['intermediate_size']:,}",
        "vocab": f"{cfg['vocab_size']:,}",
        "n_labels": str(len(labels)),
        "label_list": f"O, and B- and I- for each of {len(types)} types: {', '.join(types)}",
        "n_types": str(len(types)),
        "scored_categories": ", ".join(SG_LABELS),
        "max_len": str(MAX_LEN),
        "bio_rule": BIO_RULE,
        "embedding_tensor_names": ", ".join(n for n in names if "embeddings" in n),
        "classifier_tensor_names": ", ".join(n for n in names if n.startswith("classifier.")),
        "model_dir": str(box.model),
        "practice_dir": str(box.practice),
        "work_dir": str(box.work),
        "practice_file": str(practice),
        "n_practice": f"{len(read_jsonl(practice)):,}",
        "budget": str(budget),
    }


def render(budget: int, box) -> str:
    text = TEMPLATE.read_text()
    for key, value in fields(budget, box).items():
        text = text.replace("{" + key + "}", value)
    left = set(re.findall(r"\{([a-z_]+)\}", text))
    assert not left, f"unfilled placeholders: {sorted(left)}"
    return text

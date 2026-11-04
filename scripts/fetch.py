"""Download the model, openpii SG rows, and the model's Nemotron-PII held-out split.

The model's training notebook (Negative-Star-Innovators/local-pii-detector-minilm@61bb78f)
merges Nemotron train and test, then re-splits 90/10 with seed 42. Most of Nemotron's own
test split is therefore training data; only the rebuilt 10% split is held out.

Writes data/revisions.json with each revision and the sha256 of every file written.
Safe to rerun: existing outputs are rehashed, not refetched.
"""

import hashlib
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODEL = ("Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection", "e43200abcd5e739eddd31f55929f986b2b912b6b")
OPENPII = ("ai4privacy/pii-masking-openpii-1.5m", "a785eb528e28be2693c3718a27e066970de5dadb")
NEMOTRON = ("nvidia/Nemotron-PII", "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed")

MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "README.md"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_model() -> list[Path]:
    repo, rev = MODEL
    out = DATA / "model"
    snapshot_download(repo, revision=rev, allow_patterns=MODEL_FILES, local_dir=out)
    return [out / f for f in MODEL_FILES]


def fetch_openpii_sg(split: str) -> tuple[Path, dict]:
    repo, rev = OPENPII
    out = DATA / "raw" / f"openpii_sg_{split}.jsonl"
    src = Path(hf_hub_download(repo, f"data/{split}.jsonl", repo_type="dataset", revision=rev))
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        with src.open() as fin, tmp.open("w") as fout:
            for line in fin:
                if json.loads(line).get("region") == "SG":
                    fout.write(line)
        tmp.rename(out)
    with src.open() as f:
        total = sum(1 for _ in f)
    with out.open() as f:
        sg = sum(1 for _ in f)
    return out, {"source_rows": total, "sg_rows": sg}


def fetch_nemotron_heldout() -> tuple[Path, dict]:
    repo, rev = NEMOTRON
    out = DATA / "raw" / "nemotron_heldout.parquet"
    parts = [Dataset.from_parquet(hf_hub_download(repo, f"data/{split}-00000-of-00001.parquet",
                                                  repo_type="dataset", revision=rev))
             for split in ("train", "test")]
    split = concatenate_datasets(parts).train_test_split(test_size=0.1, seed=42)
    overlap = set(split["train"]["text"]) & set(split["test"]["text"])
    assert not overlap, f"{len(overlap)} held-out texts also appear in the model's training rows"
    if not out.exists():
        split["test"].to_parquet(out)
    return out, {"model_train_rows": len(split["train"]), "heldout_rows": len(split["test"]),
                 "heldout_texts_in_model_train": len(overlap)}


def main() -> None:
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    model_files = fetch_model()
    sg = {split: fetch_openpii_sg(split) for split in ("train", "validation")}
    nemotron, nemotron_counts = fetch_nemotron_heldout()

    def hashes(paths):
        return {str(p.relative_to(ROOT)): sha256(p) for p in paths}

    revisions = {
        MODEL[0]: {"type": "model", "revision": MODEL[1], "files": hashes(model_files)},
        OPENPII[0]: {"type": "dataset", "revision": OPENPII[1], "filter": "region == 'SG'",
                     "files": hashes(p for p, _ in sg.values()),
                     "counts": {split: c for split, (_, c) in sg.items()}},
        NEMOTRON[0]: {"type": "dataset", "revision": NEMOTRON[1],
                      "split": "concatenate(train, test).train_test_split(test_size=0.1, seed=42)['test']",
                      "files": hashes([nemotron]), "counts": nemotron_counts},
    }
    (DATA / "revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
    print(json.dumps(revisions, indent=2))


if __name__ == "__main__":
    main()

"""Validate and tokenize the prepared ALFRED splits."""

import hashlib
import json

import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from scripts.prepare_alfred import Command, normalize, validate_splits
from scripts.train_ml_baselines import SPLITS, read_split

SPLIT_SIZES = dict(zip(SPLITS, (15043, 1884, 1879)))
MODELS = {
    "bert-base-uncased": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "distilbert-base-uncased": (
        "distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be",
    ),
}


def load_dataset(directory):
    splits, fingerprints = {}, {}
    for name, expected in SPLIT_SIZES.items():
        path = directory / f"{name}.csv"
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        split = splits[name] = [Command(*row) for row in zip(*read_split(path))]
        if len(split) != expected:
            raise ValueError(f"{name}: expected {expected} rows, got {len(split)}")
        if any(not row.task_id.strip() or not normalize(row.task_desc) for row in split):
            raise ValueError(f"{name} contains a blank task ID or command")
    validate_splits(splits)
    labels = sorted({row.task_type for row in splits["train"]})
    if len(labels) != 7:
        raise ValueError("Expected seven ALFRED labels")
    return splits, {label: index for index, label in enumerate(labels)}, fingerprints


def token_statistics(tokenizer, rows):
    tokens = tokenizer([row.task_desc for row in rows], truncation=False,
                       return_attention_mask=False, verbose=False)["input_ids"]
    lengths = np.array([len(ids) for ids in tokens])
    quantiles = {f"p{p}": int(np.percentile(lengths, p, method="higher"))
                 for p in (50, 90, 95, 99, 100)}
    max_length = min(512, int(tokenizer.model_max_length), (quantiles["p99"] + 7) // 8 * 8)
    values, counts = np.unique(lengths, return_counts=True)
    return {"rows": len(rows), "min": int(lengths.min()), "mean": float(lengths.mean()),
            **quantiles, "max_length": max_length, "truncated": int(sum(lengths > max_length)),
            "histogram": {int(value): int(count) for value, count in zip(values, counts)}}


def make_loader(tokenizer, rows, labels, max_length, batch_size, generator=None):
    inputs = tokenizer([row.task_desc for row in rows], truncation=True,
                       max_length=max_length, return_attention_mask=True)
    features = [
        {**{key: values[index] for key, values in inputs.items()},
         "labels": labels[row.task_type]}
        for index, row in enumerate(rows)
    ]
    pad_multiple = 8 if max_length % 8 == 0 else None
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_multiple)
    return DataLoader(features, batch_size=batch_size, shuffle=generator is not None,
                      generator=generator, collate_fn=collator)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

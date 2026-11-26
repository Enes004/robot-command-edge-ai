"""Prepare text classification splits from the ALFRED Lite trajectories."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import unicodedata

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
FIELDS = ("task_desc", "task_type", "task_id")
UNLABELED_SUBSETS = {"tests_seen", "tests_unseen"}


@dataclass(frozen=True)
class Command:
    task_desc: str
    task_type: str
    task_id: str


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", text))


def load_commands(input_dir):
    files = sorted(input_dir.rglob("traj_data.json"))
    if not files:
        raise ValueError(f"No traj_data.json files found under {input_dir}")

    commands = []
    counts = Counter(
        trajectory_files=len(files), unlabeled_trajectories_skipped=0,
        annotations=0, empty_removed=0,
    )
    for path in files:
        try:
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            if "task_type" not in trajectory and any(
                folder in UNLABELED_SUBSETS for folder in path.parent.relative_to(input_dir).parts
            ):
                counts["unlabeled_trajectories_skipped"] += 1
                continue
            task_id = trajectory["task_id"]
            task_type = trajectory["task_type"]
            annotations = trajectory["turk_annotations"]["anns"]
            if not all(isinstance(value, str) and value.strip() for value in (task_id, task_type)):
                raise ValueError("expected non-empty task_id and task_type")
            if not isinstance(annotations, list):
                raise ValueError("expected a list of annotations")
            for annotation in annotations:
                counts["annotations"] += 1
                description = annotation["task_desc"]
                if description is not None and not isinstance(description, str):
                    raise ValueError("expected task_desc to be a string or null")
                if description is None or not normalize(description):
                    counts["empty_removed"] += 1
                    continue
                commands.append(Command(description.strip(), task_type.strip(), task_id.strip()))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
    return commands, counts


def clean_and_group(commands):
    by_text = defaultdict(list)
    task_labels = {}
    for command in commands:
        previous = task_labels.setdefault(command.task_id, command.task_type)
        if previous != command.task_type:
            raise ValueError(f"Inconsistent task type for task_id {command.task_id}")
        by_text[normalize(command.task_desc)].append(command)

    # A shared same-label command connects all its source tasks, even after deduplication.
    parent = {task_id: task_id for task_id in task_labels}

    def root(task_id):
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    conflicts = []
    duplicates = 0
    retained = []
    for key, matches in sorted(by_text.items()):
        labels = Counter(command.task_type for command in matches)
        if len(labels) > 1:
            conflicts.append({
                "normalized_text": key,
                "class_counts": dict(sorted(labels.items())),
                "task_ids": sorted({command.task_id for command in matches}),
            })
            continue

        representative = min(matches, key=lambda command: (command.task_id, command.task_desc))
        retained.append(representative)
        duplicates += len(matches) - 1
        for command in matches:
            parent[root(command.task_id)] = root(representative.task_id)

    groups = defaultdict(list)
    for command in retained:
        groups[root(command.task_id)].append(command)
    return list(groups.values()), duplicates, conflicts


def split_groups(groups):
    by_label = defaultdict(list)
    for group in groups:
        by_label[group[0].task_type].append(group)

    rng = random.Random(SEED)
    splits = {name: [] for name in SPLIT_RATIOS}
    for label in sorted(by_label):
        class_groups = by_label[label]
        rng.shuffle(class_groups)
        class_groups.sort(key=len, reverse=True)
        total = sum(map(len, class_groups))
        remaining = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
        for group in class_groups:
            name = max(remaining, key=remaining.get)
            splits[name].extend(group)
            remaining[name] -= len(group)
    return splits


def validate_splits(splits):
    task_splits = {}
    seen_texts = set()
    labels = {command.task_type for rows in splits.values() for command in rows}
    for name, rows in splits.items():
        if not rows or {command.task_type for command in rows} != labels:
            raise ValueError(f"{name} does not contain every class")
        for command in rows:
            if task_splits.setdefault(command.task_id, name) != name:
                raise ValueError(f"Task ID leakage: {command.task_id}")
            key = normalize(command.task_desc)
            if key in seen_texts:
                raise ValueError(f"Duplicate normalized text: {key}")
            seen_texts.add(key)


def plot_splits(splits, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_counts = {name: Counter(row.task_type for row in rows) for name, rows in splits.items()}
    labels = sorted(class_counts["train"])
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    totals = [0] * len(labels)
    for name, counts in class_counts.items():
        values = [counts[label] for label in labels]
        ax.barh(labels, values, left=totals, label=name)
        totals = [total + value for total, value in zip(totals, values)]
    ax.set(xlabel="Commands", title="ALFRED command counts by task type")
    ax.legend()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    lengths = {name: [len(normalize(row.task_desc).split()) for row in rows]
               for name, rows in splits.items()}
    bins = [length + 0.5 for length in range(max(map(max, lengths.values())) + 1)]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for name, values in lengths.items():
        ax.hist(values, bins=bins, weights=[1 / len(values)] * len(values),
                histtype="step", linewidth=2, label=name)
    ax.set(xlabel="Words per command", ylabel="Share within split",
           title="ALFRED command lengths")
    ax.legend()
    fig.savefig(output_dir / "command_lengths.png", dpi=160)
    plt.close(fig)


def prepare(input_dir, output_dir, minimum=1000):
    commands, counts = load_commands(input_dir)
    groups, duplicates, conflicts = clean_and_group(commands)
    usable = sum(map(len, groups))
    if usable < minimum:
        raise ValueError(f"Expected at least {minimum} usable commands, found {usable}")

    splits = split_groups(groups)
    validate_splits(splits)
    counts.update(
        duplicate_rows_removed=duplicates,
        conflicting_texts=len(conflicts),
        conflicting_rows_removed=sum(sum(entry["class_counts"].values()) for entry in conflicts),
        usable=usable,
    )
    report = {
        "seed": SEED,
        "target_ratios": SPLIT_RATIOS,
        "counts": dict(counts),
        "classes": dict(sorted(Counter(row.task_type for rows in splits.values() for row in rows).items())),
        "splits": {
            name: {
                "rows": len(rows),
                "task_ids": len({row.task_id for row in rows}),
                "classes": dict(sorted(Counter(row.task_type for row in rows).items())),
            }
            for name, rows in splits.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(FIELDS)
            writer.writerows((row.task_desc, row.task_type, row.task_id) for row in rows)
    for name, payload in (("summary", report), ("label_conflicts", conflicts)):
        (output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    plot_splits(splits, output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/json_2.1.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

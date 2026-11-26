import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_alfred import (
    Command,
    clean_and_group,
    load_commands,
    normalize,
    prepare,
    split_groups,
    validate_splits,
)


def write_trajectory(root, task_id, task_type, descriptions):
    path = root / task_type / task_id / "traj_data.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "task_id": task_id,
        "task_type": task_type,
        "turk_annotations": {
            "anns": [{"task_desc": desc, "high_descs": ["Ignored"]} for desc in descriptions]
        },
    }), encoding="utf-8")
    return path


class DatasetTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize("  ＰＩＣＫ\tUP, the_cup! "), "pick up the cup")
        self.assertEqual(normalize("…!"), "")

    def test_extract_and_filter_empty_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trajectory(root, "trial1", "pick", [" Pick a cup! ", "  ", "…", None])
            commands, counts = load_commands(root)
        self.assertEqual(commands, [Command("Pick a cup!", "pick", "trial1")])
        self.assertEqual(counts, {
            "trajectory_files": 1, "unlabeled_trajectories_skipped": 0,
            "annotations": 4, "empty_removed": 3,
        })

    def test_unlabeled_official_tests_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_trajectory(root / "tests_seen", "trial1", "pick", ["Pick a cup."])
            trajectory = json.loads(path.read_text())
            del trajectory["task_type"]
            path.write_text(json.dumps(trajectory))
            commands, counts = load_commands(root)
            self.assertEqual(commands, [])
            self.assertEqual(counts["unlabeled_trajectories_skipped"], 1)
            path.rename(root / "traj_data.json")
            with self.assertRaisesRegex(ValueError, "task_type"):
                load_commands(root)

    def test_schema_error_includes_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trajectory(root, "trial1", "pick", [42])
            with self.assertRaisesRegex(ValueError, "traj_data.json: expected task_desc"):
                load_commands(root)

    def test_deduplication_connects_tasks_transitively_and_removes_conflicts(self):
        groups, duplicates, conflicts = clean_and_group([
            Command("Alpha!", "pick", "a"),
            Command("alpha", "pick", "b"),
            Command("Beta", "pick", "b"),
            Command("BETA", "pick", "c"),
            Command("Gamma", "pick", "c"),
            Command("Ambiguous", "pick", "c"),
            Command("ambiguous!", "heat", "d"),
        ])
        self.assertEqual(duplicates, 2)
        self.assertEqual([[row.task_id for row in group] for group in groups], [["a", "b", "c"]])
        self.assertEqual(conflicts, [{
            "normalized_text": "ambiguous",
            "class_counts": {"heat": 1, "pick": 1},
            "task_ids": ["c", "d"],
        }])

    def test_inconsistent_task_id_label_fails(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent task type"):
            clean_and_group([Command("Alpha", "pick", "a"), Command("Beta", "heat", "a")])

    def test_split_is_deterministic_grouped_and_class_balanced(self):
        groups = [
            [Command(f"{label} {index} {suffix}", label, f"{label}-{index}")
             for suffix in ("a", "b")]
            for label in ("heat", "pick")
            for index in range(10)
        ]
        splits = split_groups(groups)
        self.assertEqual(splits, split_groups(groups))
        self.assertEqual([len(rows) for rows in splits.values()], [32, 4, 4])
        validate_splits(splits)
        for rows in splits.values():
            self.assertEqual(sum(row.task_type == "heat" for row in rows), len(rows) // 2)

    def test_validation_detects_both_leakage_types(self):
        first = Command("Pick a cup.", "pick", "a")
        for second, error in (
            (Command("Pick a plate", "pick", "a"), "Task ID leakage"),
            (Command("PICK A CUP!", "pick", "b"), "Duplicate normalized text"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                validate_splits({"train": [first], "test": [second]})

    def test_prepare_writes_all_artifacts_and_enforces_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "processed"
            for index in range(10):
                write_trajectory(root, str(index), "pick", [f"Pick object {index}."])
            with self.assertRaisesRegex(ValueError, "at least 1000"):
                prepare(root, output)
            self.assertFalse(output.exists())
            report = prepare(root, output, minimum=10)
            self.assertEqual(report["counts"]["usable"], 10)
            self.assertEqual(json.loads((output / "summary.json").read_text()), report)
            self.assertEqual(json.loads((output / "label_conflicts.json").read_text()), [])
            for name, split in report["splits"].items():
                with (output / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, ["task_desc", "task_type", "task_id"])
                    self.assertEqual(len(list(reader)), split["rows"])
            for name in ("class_distribution", "command_lengths"):
                self.assertTrue((output / f"{name}.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

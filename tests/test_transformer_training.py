import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.train_transformers import fit_epoch
from scripts.transformer_data import load_dataset, token_statistics


class TransformerDataTests(unittest.TestCase):
    def test_token_statistics_include_special_tokens_and_round_up(self):
        def tokenizer(texts, **kwargs):
            self.assertFalse(kwargs["truncation"])
            return {"input_ids": [[0] * length for length in (3, 9, 17)]}

        rows = [SimpleNamespace(task_desc="command")] * 3

        class Tokenizer:
            model_max_length = 512

            def __call__(self, texts, **kwargs):
                return tokenizer(texts, **kwargs)

        stats = token_statistics(Tokenizer(), rows)
        self.assertEqual(stats["max_length"], 24)
        self.assertEqual(stats["histogram"], {3: 1, 9: 1, 17: 1})

    def test_leakage_and_baseline_label_order(self):
        sizes = dict.fromkeys(("train", "validation", "test"), 7)
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.transformer_data.SPLIT_SIZES", sizes
        ):
            root = Path(directory)

            def write_split(name, first_task=None, first_text=None):
                with (root / f"{name}.csv").open("w", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(("task_desc", "task_type", "task_id"))
                    for index in reversed(range(7)):
                        writer.writerow((
                            first_text if index == 0 and first_text else f"{name} {index}",
                            f"label_{index}",
                            first_task if index == 0 and first_task else f"{name}-{index}",
                        ))

            for name in sizes:
                write_split(name)
            _, labels, hashes = load_dataset(root)
            self.assertEqual(labels, {f"label_{index}": index for index in range(7)})
            self.assertEqual(len(set(hashes.values())), 3)

            write_split("test", first_task="train-0")
            with self.assertRaisesRegex(ValueError, "Task ID leakage"):
                load_dataset(root)
            write_split("test", first_text=" TRAIN, 0! ")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized text"):
                load_dataset(root)


class GradientAccumulationTests(unittest.TestCase):
    def test_unequal_microbatches_match_one_weighted_batch(self):
        class Classifier(torch.nn.Linear):
            def forward(self, features, labels):
                logits = super().forward(features)
                return SimpleNamespace(loss=F.cross_entropy(logits, labels))

        torch.manual_seed(42)
        actual, reference = Classifier(2, 2), Classifier(2, 2)
        reference.load_state_dict(actual.state_dict())
        features = torch.randn(3, 2)
        labels = torch.tensor([0, 1, 1])
        loader = torch.utils.data.DataLoader(
            [{"features": x, "labels": y} for x, y in zip(features, labels)], batch_size=2
        )
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.1)
        loss = fit_epoch(actual, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                         torch.device("cpu"), None, 3)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        reference_loss = reference(features, labels).loss
        reference_loss.backward()
        reference_optimizer.step()
        self.assertAlmostEqual(loss, reference_loss.item(), places=6)
        for actual_parameter, reference_parameter in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_parameter, reference_parameter)


if __name__ == "__main__":
    unittest.main()

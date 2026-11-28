import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from scripts.train_ml_baselines import make_pipeline, run


class BaselineTests(unittest.TestCase):
    def test_tfidf_retains_numbers_and_one_character_tokens(self):
        pipeline = make_pipeline("logistic_regression", 1, None)
        analyzer = pipeline["tfidf"].build_analyzer()
        self.assertEqual(analyzer("A 2 Boxes"), ["a", "2", "boxes", "a 2", "2 boxes"])

    def test_full_run_selects_on_validation_and_exports_train_only_pipelines(self):
        terms = ("lamp", "table", "bowl", "sink", "fridge", "oven", "pairs")
        labels = [f"class_{index}" for index in range(len(terms))]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation", "test"):
                with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(("task_desc", "task_type", "task_id"))
                    for index, (label, term) in enumerate(zip(labels, terms)):
                        for repeat in range(2):
                            writer.writerow((f"{term} {split} {repeat}", label,
                                             f"{split}-{index}-{repeat}"))
            output = root / "results"
            summary = run(root, output)

            with (output / "hyperparameters.csv").open(newline="", encoding="utf-8") as file:
                trials = list(csv.DictReader(file))
            with (output / "test_predictions.csv").open(newline="", encoding="utf-8") as file:
                predictions = list(csv.DictReader(file))
            self.assertEqual(len(trials), 12)
            self.assertEqual(len(predictions), 14)
            self.assertEqual(summary["label_order"], labels)

            for model, result in summary["models"].items():
                with self.subTest(model=model):
                    model_trials = [trial for trial in trials if trial["model"] == model]
                    self.assertEqual({(float(row["C"]), row["class_weight"])
                                      for row in model_trials},
                                     {(c, w) for c in (0.1, 1, 10) for w in ("None", "balanced")})
                    chosen = max(model_trials, key=lambda row: float(row["validation_macro_f1"]))
                    self.assertEqual([row for row in model_trials if row["selected"] == "True"],
                                     [chosen])
                    pipeline = joblib.load(output / f"{model}.joblib")
                    self.assertIsInstance(pipeline, Pipeline)
                    self.assertEqual(pipeline["classifier"].C, float(chosen["C"]))
                    vocabulary = pipeline["tfidf"].vocabulary_
                    self.assertIn("train", vocabulary)
                    self.assertNotIn("validation", vocabulary)
                    self.assertNotIn("test", vocabulary)
                    self.assertEqual(result["matrix_dimensions"]["test"], (14, len(vocabulary)))
                    self.assertEqual(sum(map(sum, result["confusion_raw"])), 14)
                    np.testing.assert_allclose(np.sum(result["confusion_normalized"], axis=1), 1)
                    self.assertEqual(set(result["per_class"]), set(labels))
                    self.assertEqual(pipeline.predict(["lamp test 0"])[0], predictions[0][model])
                    for suffix in ("confusion_raw.png", "confusion_normalized.png", "top_features.csv"):
                        self.assertGreater((output / f"{model}_{suffix}").stat().st_size, 0)

            for name in ("metrics.json", "test_model_comparison.png",
                         "validation_model_comparison.png"):
                self.assertGreater((output / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

"""Train and evaluate traditional classifiers on the prepared ALFRED splits."""

import argparse
import csv
import json
import platform
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SPLITS = ("train", "validation", "test")
MODELS = {
    "logistic_regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
    "linear_svc": LinearSVC(loss="squared_hinge", random_state=42),
}
C_VALUES = (0.1, 1, 10)
CLASS_WEIGHTS = (None, "balanced")


def read_split(path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return ([row["task_desc"] for row in rows], [row["task_type"] for row in rows],
            [row["task_id"] for row in rows])


def make_pipeline(model_name, c, class_weight):
    # With lbfgs and seven classes, scikit-learn 1.7 uses multinomial loss.
    classifier = clone(MODELS[model_name]).set_params(C=c, class_weight=class_weight)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), token_pattern=r"\b\w+\b",
            stop_words=None, use_idf=True, smooth_idf=True, norm="l2",
        )),
        ("classifier", classifier),
    ])


def select_pipeline(model_name, train, validation, labels):
    trials = []
    best_score = -1
    for c, class_weight in product(C_VALUES, CLASS_WEIGHTS):
        pipeline = make_pipeline(model_name, c, class_weight)
        start = perf_counter()
        pipeline.fit(train[0], train[1])
        fit_seconds = perf_counter() - start
        score = f1_score(validation[1], pipeline.predict(validation[0]),
                         labels=labels, average="macro", zero_division=0)
        trial = {
            "model": model_name, "C": c, "class_weight": class_weight,
            "validation_macro_f1": score, "fit_seconds": fit_seconds,
        }
        trials.append(trial)
        if score > best_score:
            best_score = score
            best = (pipeline, trial)
    return best, trials


def evaluate(truth, predictions, labels):
    report = classification_report(truth, predictions, labels=labels,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }, {label: report[label] for label in labels}


def top_features(pipeline):
    terms = pipeline["tfidf"].get_feature_names_out()
    classifier = pipeline["classifier"]
    return [
        {"class": label, "direction": direction, "rank": rank,
         "feature": terms[index], "weight": weights[index]}
        for label, weights in zip(classifier.classes_, classifier.coef_)
        for direction, order in (("highest", -weights), ("lowest", weights))
        for rank, index in enumerate(np.argsort(order, kind="stable")[:10], start=1)
    ]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_plots(output, model_name, labels, raw, normalized):
    for name, matrix, fmt in (("raw", raw, "d"), ("normalized", normalized, ".2f")):
        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=ax, cmap="Blues", xticks_rotation=35, values_format=fmt,
        )
        ax.set_title(f"{model_name} | Test confusion matrix ({name})")
        fig.savefig(output / f"{model_name}_confusion_{name}.png", dpi=150)
        plt.close(fig)


def save_comparison_plots(output, results, trials):
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    scores = np.arange(len(next(iter(results.values()))["test_metrics"]))
    for index, (model, result) in enumerate(results.items()):
        metrics = result["test_metrics"]
        ax.bar(scores + (index - 0.5) * 0.35, list(metrics.values()), width=0.35, label=model)
    ax.set(xticks=scores, xticklabels=[key.replace("_", " ") for key in metrics],
           ylim=(0, 1.06), ylabel="Score", title="Selected models on the test split")
    ax.legend()
    fig.savefig(output / "test_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for model, weight in product(results, CLASS_WEIGHTS):
        scores = [trial["validation_macro_f1"] for trial in trials
                  if trial["model"] == model and trial["class_weight"] == weight]
        ax.plot(C_VALUES, scores, marker="o", label=f"{model}, {weight}")
    ax.set(xscale="log", xticks=C_VALUES, xticklabels=C_VALUES, xlabel="C",
           ylabel="Validation Macro F1", title="Hyperparameter comparison")
    ax.legend()
    fig.savefig(output / "validation_model_comparison.png", dpi=150)
    plt.close(fig)


def run(input_dir, output):
    train, validation, test = (read_split(input_dir / f"{name}.csv") for name in SPLITS)
    labels = sorted(set(train[1]))
    if len(labels) != 7 or any(set(split[1]) != set(labels) for split in (validation, test)):
        raise ValueError("Each ALFRED split must contain the same seven task labels.")

    output.mkdir(parents=True, exist_ok=True)
    selected, trials = {}, []
    # Small linear models avoid BLAS oversubscription and use comparable fit timing.
    with threadpool_limits(limits=1):
        for model_name in MODELS:
            selected[model_name], model_trials = select_pipeline(
                model_name, train, validation, labels,
            )
            trials.extend(model_trials)

    results = {}
    prediction_rows = [
        {"task_id": task_id, "task_type": label}
        for task_id, label in zip(test[2], test[1])
    ]
    for model_name, (pipeline, trial) in selected.items():
        predictions = pipeline.predict(test[0])
        for row, prediction in zip(prediction_rows, predictions):
            row[model_name] = prediction
        metrics, per_class = evaluate(test[1], predictions, labels)
        raw = confusion_matrix(test[1], predictions, labels=labels)
        normalized = confusion_matrix(test[1], predictions, labels=labels, normalize="true")
        vectorizer = pipeline["tfidf"]
        results[model_name] = {
            "selected": trial,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "matrix_dimensions": {
                name: vectorizer.transform(split[0]).shape
                for name, split in zip(SPLITS, (train, validation, test))
            },
            "test_metrics": metrics,
            "per_class": per_class,
            "confusion_raw": raw.tolist(),
            "confusion_normalized": normalized.tolist(),
        }
        joblib.dump(pipeline, output / f"{model_name}.joblib")
        write_csv(output / f"{model_name}_top_features.csv", top_features(pipeline))
        save_confusion_plots(output, model_name, labels, raw, normalized)

    write_csv(output / "hyperparameters.csv", [
        {**trial, "class_weight": trial["class_weight"] or "None",
         "selected": trial == selected[trial["model"]][1]}
        for trial in trials
    ])
    write_csv(output / "test_predictions.csv", prediction_rows)
    save_comparison_plots(output, results, trials)
    summary = {
        "python": platform.python_version(), "architecture": platform.machine(),
        "dependencies": {
            package: version(package)
            for package in ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl",
                            "cloudpickle", "matplotlib")
        },
        "label_order": labels, "models": results,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml-baselines"))
    args = parser.parse_args()
    summary = run(args.input_dir, args.output_dir)
    for model, result in summary["models"].items():
        print(f"{model} test Macro F1 {result['test_metrics']['macro_f1']:.6f}")
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()

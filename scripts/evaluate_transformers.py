"""Evaluate selected ALFRED checkpoints and compare with the existing ML pipelines."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.train_ml_baselines import MODELS as BASELINES
from scripts.train_ml_baselines import evaluate, plt, write_csv
from scripts.train_transformers import predict
from scripts.transformer_data import MODELS, load_dataset, make_loader, write_json


def save_figure(fig, path):
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_history(output, history):
    epochs = [row["epoch"] for row in history]
    for filename, title, fields in (
        ("loss", "Cross-entropy loss", {"train_loss": "Train", "validation_loss": "Validation"}),
        ("validation_accuracy", "Validation accuracy", {"validation_accuracy": "Accuracy"}),
        ("validation_macro_f1", "Validation Macro F1", {"validation_macro_f1": "Macro F1"}),
    ):
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for key, label in fields.items():
            ax.plot(epochs, [row[key] for row in history], marker="o", label=label)
        ax.set(xlabel="Epoch", ylabel=title, xticks=epochs, title=title)
        ax.legend()
        save_figure(fig, output / f"{filename}.png")


def score(output, labels, truth, predictions):
    metrics, per_class = evaluate(truth, predictions, labels)
    result = {"test_metrics": metrics, "per_class": per_class}
    write_csv(output / "test_metrics.csv", [metrics])
    write_csv(output / "per_class.csv", [
        {"label": label, **values} for label, values in per_class.items()
    ])
    for name, mode, fmt in (("raw", None, "d"), ("normalized", "true", ".2f")):
        key = f"confusion_{name}"
        matrix = confusion_matrix(truth, predictions, labels=labels, normalize=mode)
        result[key] = matrix.tolist()
        write_csv(output / f"{key}.csv", [
            {"true_label": label, **dict(zip(labels, row))} for label, row in zip(labels, matrix)
        ])
        fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=ax, cmap="Blues", xticks_rotation=40, values_format=fmt)
        ax.set_title(f"{output.name} | Test confusion matrix ({name})")
        save_figure(fig, output / f"{key}.png")
    return result


def evaluate_checkpoint(directory, test, labels, fingerprints, batch_size, device):
    checkpoint = directory / "best_model"
    config = json.loads((checkpoint / "training_config.json").read_text())
    summary = json.loads((directory / "training_summary.json").read_text())
    if config["smoke"] or config["split_sha256"] != fingerprints:
        raise ValueError(f"Expected a full checkpoint trained on these ALFRED splits: {directory}")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    if model.config.label2id != labels:
        raise ValueError(f"Label mapping mismatch: {directory}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    loader = make_loader(tokenizer, test, labels, config["max_length"], batch_size)
    loss, _, predicted_ids = predict(model, loader, device)
    predicted = [model.config.id2label[index] for index in predicted_ids]
    result = score(directory, list(labels), [row.task_type for row in test], predicted)
    result.update(summary, test_loss=loss, evaluation_device=str(device), split_sha256=fingerprints)
    write_json(checkpoint / "evaluation.json", result)
    plot_history(directory, summary["history"])
    return result, predicted


def plot_comparison(output, filename, rows, title):
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    positions = np.arange(len(rows[0]) - 1)
    width = 0.8 / len(rows)
    for index, row in enumerate(rows):
        ax.bar(positions + (index - (len(rows) - 1) / 2) * width,
               list(row.values())[1:], width=width, label=row["model"])
    ax.set(xticks=positions, xticklabels=[key.replace("_", " ") for key in list(rows[0])[1:]],
           ylim=(0, 1.05), ylabel="Score", title=title)
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(loc="lower right")
    save_figure(fig, output / filename)


def run(args):
    splits, labels, fingerprints = load_dataset(args.input_dir)
    test = splits["test"]
    truth = [row.task_type for row in test]
    predictions = [{"task_id": row.task_id, "task_type": row.task_type} for row in test]
    output = args.output_dir / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    baselines = json.loads((args.baseline_dir / "metrics.json").read_text())
    if baselines["label_order"] != list(labels):
        raise ValueError("Baseline label order does not match the transformer label mapping")
    for name in (*BASELINES, *args.models):
        if name in BASELINES:
            pipeline_path = args.baseline_dir / f"{name}.joblib"
            pipeline = joblib.load(pipeline_path)
            predicted = pipeline.predict([row.task_desc for row in test])
            directory = output / name
            directory.mkdir(exist_ok=True)
            result = score(directory, list(labels), truth, predicted)
            selected = baselines["models"][name]["selected"]
            result.update(validation_macro_f1=selected["validation_macro_f1"],
                          training_seconds=selected["fit_seconds"], weight_bytes=pipeline_path.stat().st_size,
                          parameter_count=None)
        else:
            result, predicted = evaluate_checkpoint(
                args.output_dir / "full" / name, test, labels, fingerprints, args.batch_size, device)
        results[name] = result
        for row, prediction in zip(predictions, predicted):
            row[name] = prediction

    performance = [{"model": name, **res["test_metrics"]} for name, res in results.items()]
    comparison = [
        {**row, **{key: results[row["model"]][key] for key in (
            "validation_macro_f1", "training_seconds", "parameter_count", "weight_bytes",
        )}} for row in performance
    ]
    class_f1 = [
        {"model": name, **{label: values["f1-score"] for label, values in res["per_class"].items()}}
        for name, res in results.items()
    ]
    write_json(output / "metrics.json", {
        "label_mapping": labels, "split_sha256": fingerprints, "models": results,
    })
    write_csv(output / "test_predictions.csv", predictions)
    write_csv(output / "performance.csv", comparison)
    write_csv(output / "per_class_f1.csv", class_f1)
    plot_comparison(output, "performance.png", performance, "Test set performance")
    plot_comparison(output, "per_class_f1.png", class_f1, "Test F1 by class")
    print(f"Comparison written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformers"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/ml-baselines"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="*", choices=MODELS, default=tuple(MODELS),
                        help="Transformer checkpoints to evaluate. Pass without names for baseline-only results.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

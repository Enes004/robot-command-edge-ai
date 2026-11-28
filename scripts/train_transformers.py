"""Fine-tune the ALFRED transformers and select checkpoints on validation Macro F1."""

import argparse
import platform
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from scripts.train_ml_baselines import write_csv
from scripts.transformer_data import MODELS, load_dataset, make_loader, token_statistics, write_json


def forward(model, batch, device, precision):
    inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.autocast(device_type=device.type, dtype=precision, enabled=precision is not None):
        result = model(**inputs)
    return result


@torch.inference_mode()
def predict(model, loader, device, precision=None):
    model.eval()
    loss_sum, truth, predictions = 0.0, [], []
    for batch in loader:
        result = forward(model, batch, device, precision)
        loss_sum += result.loss.item() * len(batch["labels"])
        truth.extend(batch["labels"].tolist())
        predictions.extend(result.logits.argmax(dim=-1).tolist())
    return loss_sum / len(truth), truth, predictions


def fit_epoch(model, loader, optimizer, scaler, device, precision, accumulation, log_interval=0):
    model.train()
    total_loss = torch.zeros((), device=device)
    batches = iter(loader)
    step = 0
    while group := list(islice(batches, accumulation)):
        group_size = sum(len(batch["labels"]) for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            result = forward(model, batch, device, precision)
            batch_size = len(batch["labels"])
            scaler.scale(result.loss * (batch_size / group_size)).backward()
            total_loss += result.loss.detach() * batch_size
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if log_interval and step % log_interval == 0:
            print(f"Optimizer step {step}/{(len(loader) + accumulation - 1) // accumulation}",
                  flush=True)
    return total_loss.item() / len(loader.dataset)


def smoke_sample(rows, size, seed):
    sample, _ = train_test_split(rows, train_size=size, random_state=seed,
                                 stratify=[row.task_type for row in rows])
    return sample


def run(args):
    if min(args.batch_size, args.accumulation, args.epochs) < 1 or args.learning_rate <= 0:
        raise ValueError("Batch size, accumulation, epochs and learning rate must be positive")
    splits, labels, fingerprints = load_dataset(args.input_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but PyTorch cannot access a CUDA device")
    precision = None
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    set_seed(args.seed)
    output = args.output_dir / ("smoke" if args.smoke else "full") / args.model
    output.mkdir(parents=True, exist_ok=False)
    source, revision = MODELS[args.model] if args.source is None else (args.source, None)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    stats = token_statistics(tokenizer, splits["train"])
    write_json(output / "token_lengths.json", stats)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, revision=revision, num_labels=len(labels), label2id=labels,
        id2label={index: label for label, index in labels.items()},
        problem_type="single_label_classification",
    ).to(device)
    if model.config.model_type != args.model.split("-")[0]:
        raise ValueError(f"Expected {args.model}, got model type {model.config.model_type}")

    train, validation = splits["train"], splits["validation"]
    if args.smoke:
        train = smoke_sample(train, 64, args.seed)
        validation = smoke_sample(validation, 32, args.seed)
    train_loader = make_loader(tokenizer, train, labels, stats["max_length"], args.batch_size,
                               torch.Generator().manual_seed(args.seed))
    val_loader = make_loader(tokenizer, validation, labels, stats["max_length"], args.batch_size)
    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01,
                                  fused=fused)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == torch.float16)
    config = {
        "model": args.model, "source": str(source), "revision": revision,
        "smoke": args.smoke, "split_sha256": fingerprints, "seed": args.seed,
        "epochs": 1 if args.smoke else args.epochs, "optimizer": "AdamW", "weight_decay": 0.01,
        "fused_adamw": fused, "full_fine_tuning": True,
        "learning_rate": args.learning_rate, "loss": "CrossEntropyLoss",
        "train_rows": len(train), "validation_rows": len(validation),
        "batch_size": args.batch_size, "gradient_accumulation": args.accumulation,
        "effective_batch_size": args.batch_size * args.accumulation,
        "batch_adjustment_reason": args.batch_reason, "max_length": stats["max_length"],
        "checkpoint_metric": "validation_macro_f1", "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda, "precision": str(precision or torch.float32),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
            "numpy", "scikit-learn", "matplotlib",
        )},
    }
    write_json(output / "training_config.json", config)
    best_score, history = -1, []
    checkpoint = output / "best_model"
    start = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        print(f"{args.model} epoch {epoch}/{config['epochs']} on {device}", flush=True)
        train_loss = fit_epoch(model, train_loader, optimizer, scaler, device, precision,
                               args.accumulation, log_interval=100)
        val_loss, truth, predictions = predict(model, val_loader, device, precision)
        score = f1_score(truth, predictions, labels=list(labels.values()), average="macro",
                         zero_division=0)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                  "validation_accuracy": accuracy_score(truth, predictions),
                  "validation_macro_f1": score}
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint / "label_mapping.json", labels)
            write_json(checkpoint / "training_config.json", config)
        history.append(record)
        write_csv(output / "history.csv", history)
        print(record, flush=True)
    write_json(output / "training_summary.json", {
        "best_epoch": best_epoch, "validation_macro_f1": best_score,
        "training_seconds": perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "weight_bytes": sum(path.stat().st_size for path in checkpoint.glob("*.safetensors")),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "history": history,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--source", help="Optional local directory containing pretrained model files")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformers"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--batch-reason", default="Task 03 defaults")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

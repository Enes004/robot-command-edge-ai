# Deployment handover for Mertcan

The ZIP contains the two fitted scikit-learn pipelines, the best BERT and
DistilBERT checkpoints, the processed ALFRED test split and the measured
reports. Extract it and run the commands from the extracted directory.

## Package contents

| Location | Contents |
| --- | --- |
| `models/ml/` | `logistic_regression.joblib` and `linear_svc.joblib`, each containing its fitted TF-IDF vectorizer and classifier |
| `models/bert-base-uncased/best_model/` | BERT weights, model config, tokenizer, label mapping, training config and evaluation |
| `models/distilbert-base-uncased/best_model/` | DistilBERT weights and the same metadata |
| `data/processed/test.csv` | 1,879 rows with `task_desc`, `task_type` and `task_id` |
| `results/` | Measured CSV and JSON reports, training curves, confusion matrices and comparison plots |
| `requirements-*.txt` | Recorded dependency pins |

## Environment and inference

The reference environment used Python 3.10.12, scikit-learn 1.7.2,
PyTorch 2.9.1 and Transformers 5.20.2. The pins in the package describe the
tested x86-64 environment. For Raspberry Pi 5, use 64-bit Raspberry Pi OS
and check wheel availability for the target Python version and architecture.
Record any version changes, especially for scikit-learn and joblib when loading
serialized pipelines.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-transformers.txt
```

The classical pipelines take raw strings. Their fitted word unigram/bigram
TF-IDF vectorizers handle tokenization and lowercasing. Load joblib files only
from a trusted source.

```python
from pathlib import Path
import joblib

commands = ["Put the book on the shelf."]
for name in ("logistic_regression", "linear_svc"):
    pipeline = joblib.load(Path("models/ml") / f"{name}.joblib")
    print(name, pipeline.predict(commands).tolist())
```

The transformers use the bundled uncased tokenizer, truncation at the saved
`max_length` of 24, dynamic padding and attention masks. Feed the original
command strings and use `id2label` to decode the highest logit. `task_id`
serves as provenance, and `task_type` is the reference label for evaluation.

```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

commands = ["Put the book on the shelf."]
for name in ("bert-base-uncased", "distilbert-base-uncased"):
    path = Path("models") / name / "best_model"
    config = json.loads((path / "training_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    model.eval()
    batch = tokenizer(commands, padding=True, truncation=True,
                      max_length=config["max_length"], return_attention_mask=True,
                      return_tensors="pt")
    with torch.inference_mode():
        ids = model(**batch).logits.argmax(dim=-1).tolist()
    print(name, [model.config.id2label[index] for index in ids])
```

## Expected evaluation metrics

These are the recorded results on the bundled test CSV. Use all seven classes
when calculating macro averages. The complete scores and per-class reports
are in `results/comparison/metrics.json` and the model directories.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.93081 | 0.93546 | 0.93402 | 0.93440 | 0.93136 |
| Linear SVM | 0.93294 | 0.93703 | 0.93563 | 0.93602 | 0.93346 |
| BERT | 0.93986 | 0.94307 | 0.94244 | 0.94215 | 0.94048 |
| DistilBERT | 0.93933 | 0.94402 | 0.94081 | 0.94164 | 0.93998 |

## Remaining deployment work

1. Export both selected transformer checkpoints to FP32 ONNX with dynamic
   batch and sequence dimensions. Keep the model's tokenizer and label
   mapping with each export. Check logits and predicted labels against
   PyTorch on the packaged test split.
2. Produce INT8 variants and measure test Macro F1, per-class F1 and model
   size against the corresponding FP32 exports.
3. Benchmark the two ML pipelines and each available transformer variant on
   Raspberry Pi 5. Record the OS, runtime versions, thread count and cooling
   setup. Use batch size one for latency, separate warm and cold runs, and
   report median and p95 end-to-end latency, throughput, and peak process RSS.

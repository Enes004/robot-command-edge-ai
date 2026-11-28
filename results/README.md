# Measured experiment results

This directory contains copies of the recorded ALFRED evaluation outputs.
All four models were evaluated on the same 1,879 held-out commands.
Transformer checkpoints were selected by validation Macro F1. Both selected
checkpoints came from epoch 1.

- `comparison/` contains aggregate metrics, per-class F1 scores and
  four-model plots.
- `ml-baselines/` contains the Logistic Regression and Linear SVM reports,
  validation search results, feature weights, and confusion matrices.
- `transformers/` contains the BERT and DistilBERT reports, token lengths,
  training configuration, epoch history, and plots of loss, validation accuracy
  and validation Macro F1.

Confusion matrix rows are true labels and columns are predicted labels.
Normalized rows report fractions of each true class. Evaluation plots in the
four model directories and `comparison/` are 300 dpi. The baseline-only
validation search and comparison plots retain their original resolution.

See [HANDOVER.md](../HANDOVER.md) for model loading and the remaining Raspberry Pi 5 work.

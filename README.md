# robot-command-edge-ai

Natural language robot command classification using classical ML and Transformers, with INT8 quantization and edge deployment on Raspberry Pi 5.

## Prepare the ALFRED dataset

Run from the repository root with Python 3.10 or newer, `curl`, and `7z`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-data.txt

mkdir -p data/raw
curl -fL --retry 2 -o data/raw/json_2.1.0.7z \
  https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z
sha256sum -c <<'CHECKSUM'
c2e9e4256f94ccb2451dcaf99ff26124777eda0f256110489d1f0eaf6a3f02e9  data/raw/json_2.1.0.7z
CHECKSUM
7z x data/raw/json_2.1.0.7z -odata/raw

python -m scripts.prepare_alfred
python -m unittest discover -s tests -p 'test_prepare_alfred.py' -v
```

Use `--input-dir` and `--output-dir` to override the defaults. The tests use small local fixtures and run independently of the download.

## Source and schema

The official ALFRED 2.1.0 Lite archive contains trajectory JSON files. The checksum above records the archive downloaded from the published S3 endpoint. The upstream README lists the MIT license and links to the repository license, copyright 2019 ALFRED. The Lite archive contains only trajectory JSON files and has no separate license file. Cite Shridhar et al., *ALFRED*, CVPR 2020. [Dataset documentation](https://github.com/askforalfred/alfred/blob/master/data/README.md) · [Official download script](https://github.com/askforalfred/alfred/blob/master/data/download_data.sh) · [Upstream license](https://github.com/askforalfred/alfred/blob/master/LICENSE) · [Paper](https://arxiv.org/abs/1912.01734)

The archive contains 6,574 train, 251 validation seen, and 255 validation unseen trajectories with top-level `task_id` and `task_type`. For each trajectory, the pipeline takes the human-written goal from each `turk_annotations.anns[*].task_desc`. The 483 test seen and 488 test unseen trajectories have goal annotations and omit `task_type`. The pipeline skips them and builds project splits from the labeled trajectories.

## Cleaning and splits

- The CSV retains the original command wording with surrounding whitespace trimmed. The comparison key applies Unicode NFKC normalization, case folding, and joins letter and number runs with single spaces. Missing text values represented by JSON null and keys without words or numbers are discarded.
- Each same-label normalized text contributes one record. The representative is the smallest `task_id` in lexical order, with ties broken by the original command text. Every task linked by a shared same-label command is assigned to the same split, including transitive links.
- Every occurrence of a normalized text with conflicting labels is discarded. The conflict report lists normalized text, counts per label, and source task IDs. For example, “put a box on a table” had three annotations each for simple placement and placement with a movable receptacle. The most frequent conflict had 13 annotations for a chilled potato command, 12 labeled cooling and one labeled cleaning. The pipeline retains the source labels for the commands it keeps.
- Seed 42 breaks ties between equally sized groups. For each class, the algorithm assigns larger groups first to the split furthest below its target row count. The target is 80/10/10 by command count. It checks that each split contains every class, no task ID crosses splits, and every normalized text is unique.

## Verified results

The pipeline scanned 8,051 files, skipped 971 unlabeled trajectories, and read 22,666 labeled goal annotations. It found zero empty commands, removed 3,735 same-label duplicates, and removed 125 rows across 37 conflicting texts. The remaining 18,806 commands span seven classes.

| Task type | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| `look_at_obj_in_light` | 1,527 | 192 | 189 | 1,908 |
| `pick_and_place_simple` | 2,317 | 290 | 290 | 2,897 |
| `pick_and_place_with_movable_recep` | 2,543 | 318 | 318 | 3,179 |
| `pick_clean_then_place_in_recep` | 2,058 | 258 | 258 | 2,574 |
| `pick_cool_then_place_in_recep` | 1,971 | 247 | 246 | 2,464 |
| `pick_heat_then_place_in_recep` | 2,035 | 255 | 255 | 2,545 |
| `pick_two_obj_and_place` | 2,592 | 324 | 323 | 3,239 |
| **Total** | **15,043** | **1,884** | **1,879** | **18,806** |

## Outputs and limitations

The pipeline writes the following files under `data/processed/`.

- `train.csv`, `validation.csv`, and `test.csv` with columns `task_desc`, `task_type`, and `task_id`. Use `task_desc` as the model input and `task_type` as the target. Keep `task_id` for provenance.
- `summary.json` with cleaning counts and overall and per-split class counts, and `label_conflicts.json` with the conflict audit.
- `class_distribution.png` and `command_lengths.png`. The histogram uses the number of words in the normalized comparison key.

ALFRED covers a fixed set of simulated household tasks, so the labels and command styles represent that domain. Goal descriptions can omit actions that distinguish classes. Removing conflicting texts reduces the ambiguity present in the source, and distinct normalized texts can still be near-paraphrases. Project splits may share scenes and annotators. Use them for text classification metrics within this project. ALFRED’s official benchmark measures grounded instruction following with its own evaluation splits.

## Traditional ML baselines

From the repository root, activate the virtual environment and use the CSVs from the dataset preparation step.

```sh
python -m pip install -r requirements-ml.txt
python -m scripts.train_ml_baselines
python -m unittest discover -s tests -v
```

Use `--input-dir` to select the directory containing the three CSVs and `--output-dir` to change the artifact location. Each candidate fits word unigram and bigram TF-IDF and its classifier on the original training CSV. The vectorizer lowercases text and keeps numbers and single-character tokens. The script scores six configurations per model on validation Macro F1. Exact ties keep the first configuration in grid order (`C` ascending, then `None` before `balanced`). The winning train-fitted pipelines are evaluated on the test CSV.

### Measured results

Both pipelines learned a vocabulary of 10,097 features. The sparse train, validation and test matrices were 15,043 × 10,097, 1,884 × 10,097 and 1,879 × 10,097, respectively.

| Model | C | Class weight | Validation Macro F1 | Test accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 | Fit (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multinomial logistic regression | 10 | None | 0.93398 | 0.93081 | 0.93546 | 0.93402 | 0.93440 | 0.93136 | 0.85 |
| LinearSVC, squared hinge | 1 | Balanced | 0.93080 | 0.93294 | 0.93703 | 0.93563 | 0.93602 | 0.93346 | 0.27 |

Fit times are wall-clock measurements for each selected pipeline, including TF-IDF fitting, with numerical library thread pools limited to one thread. The run used Python 3.10.12 on x86-64. The dependency pins and recorded runtime versions provide a reference for reproducing the environment on 64-bit Raspberry Pi OS. Raspberry Pi installation, model loading and performance remain unverified.

### Generated artifacts

The default artifact directory is `artifacts/ml-baselines/`.

- `metrics.json` records the dependency versions, vocabulary size, matrix dimensions, aggregate and per-class scores, and confusion matrices with a shared label order. Matrix rows are true classes and columns are predictions. Normalized rows show the share of each true class assigned to each predicted class.
- `hyperparameters.csv` records all 12 validation scores and fit times. `test_predictions.csv` contains both models’ predicted labels in test CSV order with the source task ID and true label.
- Each model’s `*_top_features.csv` lists the ten highest and ten lowest coefficients per class. Coefficients describe model weights for tokens and token pairs.
- PNG files show the validation grid, the test model comparison and both confusion matrices per model.
- Two `.joblib` files contain complete pipelines. Load trusted artifacts with the pinned dependencies and pass raw command strings to `predict`.

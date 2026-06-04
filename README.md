# CS205 - Feature Selection with Nearest Neighbor (Project 2)

## Overview

Goal: Find good feature subsets for a **1-nearest-neighbor** classifier. Data is two-class, continuous features (class in the first column). Features get **min-max normalized**, then each subset is scored with **leave-one-out** cross-validation. Search is greedy **forward selection** (start empty, add one feature per level) or **backward elimination** (start with all features, drop one per level). The program prints the assignment-style trace so you can follow each candidate subset and accuracy.

Team: Nikhil Rao, Akshay.

## How to run

From this folder:

```bash
python3 main.py
```

1. Pick a dataset: **1** small, **2** large, **3** real (wine).
2. Pick an algorithm: **1** forward selection, **2** backward elimination, **3** test one fixed feature subset (full per-instance trace).

If you use Conda, activate your env first, then run the same command.

Dataset files live in `datasets/`. Paths are tied to `main.py`, so you do not have to `cd` into a specific folder as long as you run from the project root (or pass the usual relative layout).

## Code map

**Classes**

- `NearestNeighbor` - stores training rows, predicts by smallest squared Euclidean distance on the chosen features (feature ids are **1-indexed** like `{4,5}` in the handout).
- `Validator` - leave-one-out wrapper: hold out each instance once, train NN on the rest, count correct guesses.

**Helpers / search**

- `load_data`, `normalize_features` - read file (space-separated CS170 sets; comma-separated wine), min-max each column.
- `format_feature_set`, `get_level_text` - trace formatting for subsets and search levels.
- `forward_selection`, `backward_elimination` - nested-loop greedy search; keeps the best subset seen at any level even if accuracy drops later (warning line matches the sample output).

**Driver**

- `pick_dataset` - menu for small / large / real.
- `run_specific_feature_subset_test` - option 3 only; timed load + full LOOCV trace.
- `main()` - dataset + algorithm menus, normalize, run search, print final best subset and accuracy.

## Datasets

| Menu | File | Notes |
|------|------|--------|
| 1 Small | `datasets/CS170_Small_DataSet__17.txt` | 700 instances, **8** features |
| 2 Large | `datasets/CS170_Large_DataSet__23.txt` | 3000 instances, **18** features |
| 3 Real | `datasets/wine_data.txt` | 178 instances, **13** features (Part 2 style) |

**Subset test presets (menu 3)**

- Small: `{3, 5, 7}`
- Large: `{1, 5, 17}` - must stay at most 18; an old handout example used feature 27 on a different 27-feature file.

## Requirements

Python 3.x - stdlib only (`os`, `time`). No NumPy/pandas required.
# CS205 - Feature Selection with Nearest Neighbor (Project 2)

## Overview

Part 1 of the project: pick good feature subsets for a nearest neighbor classifier. Our data has two classes and continuous features (class label is the first column). We min-max normalize, score each subset with leave-one-out cross-validation, then run forward selection or backward elimination. The program prints the trace the assignment expects so you can see every subset and accuracy.

Team: Nikhil Rao, Akshay.

## How to run

From this folder:

```bash
python3 main.py
```

1. Pick a dataset: **1** small, **2** large, **3** real (wine)
2. Pick an algorithm: **1** forward selection, **2** backward elimination, **3** test one fixed subset (prints every instance)

If you use Conda, activate your env first, then same command.

Input files are in `datasets/`. Paths are tied to `main.py` so you can run from the project root.

## Code map

**Classes**

- `NearestNeighbor` - finds the closest training row on the selected features (numbered from 1 like `{4,5}` in the handout)
- `Validator` - leave-one-out wrapper that scores how good a subset is

**Helpers / search**

- `load_data`, `normalize_features` - read the file and scale each feature column (wine is csv, cs170 files are space separated)
- `format_feature_set`, `get_level_text` - formatting for the search trace
- `forward_selection`, `backward_elimination` - greedy nested-loop search; keeps the best subset seen at any level

**Driver**

- `pick_dataset` - small / large / real menu
- `run_specific_feature_subset_test` - option 3, full leave-one-out trace
- `main()` - ties it all together

## Datasets

| Menu | File | Notes |
|------|------|--------|
| 1 Small | `datasets/CS170_Small_DataSet__17.txt` | 700 rows, 8 features |
| 2 Large | `datasets/CS170_Large_DataSet__23.txt` | 3000 rows, 18 features |
| 3 Real | `datasets/wine_data.txt` | 178 rows, 13 features |

**Subset test presets (menu 3)**

- Small: `{3, 5, 7}`
- Large: `{1, 5, 17}` (only valid because this file has 18 features)

## Requirements

`main.py` uses Python 3 stdlib only (`os`, `time`).

## Dendrogram (Part 3)

Clustering is a separate script (needs pandas, scipy, sklearn, matplotlib):

```bash
python3 dendrogram.py
```

Uses wine from `datasets/`, saves the plot to `data/wine_dendrogram.png`, prints a cluster vs class table, and saves that to `data/wine_cluster_comparison.txt`.

import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import MinMaxScaler

# part 3 - dendrogram for wine (178 samples, good size)
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WINE_FILE = os.path.join(DATASET_FOLDER, "wine_data.txt")
OUTPUT_FILE = os.path.join(DATA_FOLDER, "wine_dendrogram.png")
COMPARISON_FILE = os.path.join(DATA_FOLDER, "wine_cluster_comparison.txt")


def load_wine_data():
    data = pd.read_csv(WINE_FILE, header=None)
    wine_classes = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    return wine_classes, features


def print_cluster_comparison(wine_classes, linked):
    # cut tree into 3 groups and see if they line up with the 3 wine cultivars
    clusters = fcluster(linked, t=3, criterion="maxclust")

    comparison = pd.crosstab(
        wine_classes,
        clusters,
        rownames=["Wine Class"],
        colnames=["Cluster"],
    )

    print("\nCluster vs actual wine class:")
    print(comparison)
    print()

    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(COMPARISON_FILE, "w") as file:
        file.write("Cluster vs actual wine class\n")
        file.write(comparison.to_string())
        file.write("\n")
    print(f"saved {COMPARISON_FILE}")


def main():
    if not os.path.isfile(WINE_FILE):
        print(f"could not find {WINE_FILE}")
        return

    wine_classes, features = load_wine_data()
    normalized = MinMaxScaler().fit_transform(features)

    linked = linkage(normalized, method="ward", metric="euclidean")
    print_cluster_comparison(wine_classes, linked)

    color_threshold = linked[-2, 2]

    plt.rcParams["lines.linewidth"] = 2.5
    fig, ax = plt.subplots(figsize=(24, 10))

    dendrogram(
        linked,
        ax=ax,
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="#555555",
    )

    fig.suptitle("Wine Dataset Hierarchical Clustering", fontsize=16, y=0.97)
    fig.text(
        0.5,
        0.925,
        "Ward linkage on min-max normalized features (178 samples)",
        ha="center",
        fontsize=13,
    )

    ax.set_xlabel("Wine samples (labels hidden for readability)", fontsize=16, labelpad=14)
    ax.set_ylabel("Merge distance", fontsize=16, labelpad=14)
    ax.tick_params(axis="both", labelsize=13)

    for line in ax.get_lines():
        line.set_linewidth(2.5)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.86)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, facecolor="white", pad_inches=0.35)
    print(f"saved {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
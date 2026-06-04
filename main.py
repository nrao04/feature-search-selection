import os
import time

# ps: paths work even if you run the program from a different folder
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

DATASETS = {
    1: ("Small dataset", "CS170_Small_DataSet__17.txt"),
    2: ("Large dataset", "CS170_Large_DataSet__23.txt"),
    3: ("Real dataset", "wine_data.txt"),
}

SUBSET_TESTS = {
    1: (1, {3, 5, 7}),
    2: (2, {1, 15, 27}),
}

# concise helpers reused later

def pick_dataset():
    
    print("\nWhich dataset do you want to use?")
    for num, (label, _) in DATASETS.items():
        print(f"    {num}) {label}")

    try:
        choice = int(input().strip())
    except ValueError:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return None

    if choice not in DATASETS:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return None

    file_path = os.path.join(DATASET_FOLDER, DATASETS[choice][1])
    if not os.path.isfile(file_path):
        print(f"Error: could not find dataset file at {file_path}")
        return None

    label, file_name = DATASETS[choice]
    print(f"Selected {label} ({file_name})")
    return file_path


def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # wine came in as csv, and files are space separated
            if "," in line:
                parts = [value.strip() for value in line.split(",") if value.strip()]
            else:
                parts = line.split()

            class_label = int(float(parts[0]))
            data.append((class_label, [float(value) for value in parts[1:]]))

    return data


def normalize_features(data):
    
    if not data or not data[0][1]:
        return data

    num_features = len(data[0][1])
    mins = [float("inf")] * num_features
    maxs = [float("-inf")] * num_features

    for class_label, features in data:
        for i, value in enumerate(features):
            if value < mins[i]:
                mins[i] = value
            # separate ifs — elif skips max on the first value in a column
            if value > maxs[i]:
                maxs[i] = value

    normalized = []
    for class_label, features in data:
        new_features = []
        for i, value in enumerate(features):
            spread = maxs[i] - mins[i]
            if spread == 0:
                new_features.append(0.0)  # column never changes, avoid divide by zero
            else:
                new_features.append((value - mins[i]) / spread)
        normalized.append((class_label, new_features))

    return normalized

# nearest neighbor classifier

class NearestNeighbor:
    
    def __init__(self):
        self.training_data = []

    def train(self, training_data):
        self.training_data = training_data

    def test(self, test_features, feature_subset):
        if not self.training_data:
            return None

        # prints features as {1,2,3} not {0,1,2}
        feature_list = sorted(feature_subset)
        test_values = [test_features[feature_id - 1] for feature_id in feature_list]

        best_distance = float("inf")
        best_label = None
        for train_label, train_features in self.training_data:
            train_values = [train_features[feature_id - 1] for feature_id in feature_list]
            distance = sum((a - b) ** 2 for a, b in zip(test_values, train_values))
            if distance < best_distance:
                best_distance = distance
                best_label = train_label

        return best_label
    
# validator wrapper (scores how good a feature subset is)

class Validator:
    def __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data

    def evaluate(self, feature_subset, show_details=True):
        if not self.data:
            return 0.0

        total = len(self.data)
        correct = 0

        for i in range(total):
            # leave one out, train on everyone except the row we're testing
            train_data = self.data[:i] + self.data[i + 1:]
            true_label, test_features = self.data[i]

            self.classifier.train(train_data)
            guess = self.classifier.test(test_features, feature_subset)
            match = guess == true_label
            correct += match

            if show_details:
                print(
                    f"Instance Id: {i}, Correct Label: {float(true_label):.1f}, "
                    f"Guessed Label: {float(guess):.1f}, Accurate: {match}"
                )

        accuracy = correct / total
        if show_details:
            print(f"\nCorrectly Classified {correct}/{total} instances.")
            print(f"Accuracy: {accuracy:.2f}")

        return accuracy

# small helpers for formatting

def format_feature_set(features):
    return "{" + ",".join(str(feature_id) for feature_id in sorted(features)) + "}"


def get_level_text(level):
    words = ["", "single", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    if level == 1:
        return "single features"
    if level <= 10:
        return f"{words[level]}-feature sets"
    return f"{level}-feature sets"

    
# start of search algorithms

def forward_selection(num_features, validator):
    current = []
    base_acc = validator.evaluate(set(current), show_details=False) * 100.0

    print(
        "Running nearest neighbor with no features (default rate), "
        f'using "leaving-one-out" evaluation, I get an accuracy of {base_acc:.1f}%'
    )
    print("\nBeginning search.")

    best_set = current.copy()
    best_acc = base_acc

    for level in range(1, num_features + 1):
        print(f"\nEvaluating {get_level_text(level)}:")
        level_best = None
        level_acc = -1.0

        for feature_id in sorted(set(range(1, num_features + 1)) - set(current)):
            candidate = current + [feature_id]
            acc = validator.evaluate(set(candidate), show_details=False) * 100.0
            print(f"Using feature(s) {format_feature_set(candidate)} accuracy is {acc:.1f}%")
            if acc > level_acc:
                level_acc = acc
                level_best = candidate

        if level_acc < best_acc:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        # track best subset from any level, not just the last one
        if level_acc > best_acc:
            best_acc = level_acc
            best_set = level_best.copy()

        print(f"Feature set {format_feature_set(level_best)} was best, accuracy is {level_acc:.1f}%")
        current = level_best

    return set(best_set), best_acc

def backward_elimination(num_features, validator):
    current = set(range(1, num_features + 1))
    base_acc = validator.evaluate(current, show_details=False) * 100.0

    print(
        f'Running nearest neighbor with all {num_features} features, using "leaving-one-out" '
        f"evaluation, I get an accuracy of {base_acc:.1f}%"
    )
    print("\nBeginning search.")

    best_set = current.copy()
    best_acc = base_acc

    for level in range(num_features - 1, 0, -1):
        print(f"\nEvaluating {get_level_text(level)}:")
        level_best = None
        level_acc = -1.0

        for feature_id in sorted(current):
            candidate = current - {feature_id}
            acc = validator.evaluate(candidate, show_details=False) * 100.0
            print(f"Using feature(s) {format_feature_set(candidate)} accuracy is {acc:.1f}%")
            if acc > level_acc:
                level_acc = acc
                level_best = candidate

        if level_acc < best_acc:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        if level_acc > best_acc:
            best_acc = level_acc
            best_set = level_best.copy()

        print(f"Feature set {format_feature_set(level_best)} was best, accuracy is {level_acc:.1f}%")
        current = level_best

    return best_set, best_acc

# formats a set of feature IDs as a string like '{1,3,5}'

def run_specific_feature_subset_test():
    print("\nTest a specific feature subset (full leave-one-out trace):")
    print("    1) Small dataset - features {3, 5, 7}")
    print("    2) Large dataset - features {1, 15, 27}")

    try:
        choice = int(input().strip())
    except ValueError:
        print("Invalid input, please enter 1 or 2.")
        return

    if choice not in SUBSET_TESTS:
        print("Invalid choice")
        return

    dataset_num, features = SUBSET_TESTS[choice]
    file_path = os.path.join(DATASET_FOLDER, DATASETS[dataset_num][1])
    if not os.path.isfile(file_path):
        print(f"Error: could not find dataset file at {file_path}")
        return

    print(f"\nReading data from {file_path}...")
    start = time.time()
    data = load_data(file_path)
    if not data:
        print("Error: No data loaded from file")
        return

    normalized = normalize_features(data)
    load_time = time.time() - start

    print(f"Time to load and normalize data: {load_time:.4f} seconds")
    print(f"Loaded {len(normalized)} instances with {len(normalized[0][1])} features")
    print("\nPerforming leave-one-out cross-validation...")

    validator = Validator(NearestNeighbor(), normalized)
    start = time.time()
    accuracy = validator.evaluate(features, show_details=True)
    val_time = time.time() - start

    print(f"\nTime for leave-one-out validation: {val_time:.4f} seconds")
    print(f"Using feature(s) {format_feature_set(features)}, accuracy is about {accuracy:.3f}")
    print(f"\nTotal time: {load_time + val_time:.4f} seconds")


def main():
    print("Welcome to Nikhil's and Akshay's Feature Selection Algorithm.")

    file_path = pick_dataset()
    if file_path is None:
        return

    print("\nType the number of the algorithm you want to run.")
    print("    1) Forward Selection")
    print("    2) Backward Elimination")
    print("    3) Test Specific Feature Subset")

    try:
        algo = int(input().strip())
    except ValueError:
        print("Invalid choice")
        return

    if algo == 3:
        run_specific_feature_subset_test()
        return
    if algo not in (1, 2):
        print("Invalid choice")
        return

    data = load_data(file_path)
    if not data:
        print("Error: No data loaded from file")
        return

    num_features = len(data[0][1])
    print(
        f"\nThis dataset has {num_features} features (not including the class attribute), "
        f"with {len(data)} instances."
    )
    print("\nPlease wait while I normalize the data... Done!\n")

    validator = Validator(NearestNeighbor(), normalize_features(data))
    if algo == 1:
        best_features, best_acc = forward_selection(num_features, validator)
    else:
        best_features, best_acc = backward_elimination(num_features, validator)

    print(
        f"\nFinished search!! The best feature subset is {format_feature_set(best_features)}, "
        f"which has an accuracy of {best_acc:.1f}%"
    )

if __name__ == "__main__":
    main()
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
            # leave one out — train on everyone except the row we're testing
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

def format_feature_set(feature_collection):
    return "{" + ",".join(str(feature_id) for feature_id in sorted(feature_collection)) + "}"


def get_level_text(level_number):
    level_words = ["", "single", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    if level_number == 1:
        return "single features"
    if level_number <= 10:
        return f"{level_words[level_number]}-feature sets"
    return f"{level_number}-feature sets"
    
# start of search algorithms
def forward_selection(total_features, validator):
    # Start empty then add one feature at a time
    current_feature_list = []
    base_accuracy = validator.evaluate(set(current_feature_list), show_details=False) * 100.0

    print(
        "Running nearest neighbor with no features (default rate), "
        f'using "leaving-one-out" evaluation, I get an accuracy of {base_accuracy:.1f}%'
    )
    print()
    print("Beginning search.")

    best_overall_list = current_feature_list.copy()
    best_overall_accuracy = base_accuracy

    for level_number in range(1, total_features + 1):
        print(f"\nEvaluating {get_level_text(level_number)}:")

        best_level_list = None
        best_level_accuracy = -1.0

        remaining_features = sorted(set(range(1, total_features + 1)) - set(current_feature_list))
        for feature_id in remaining_features:
            candidate_list = current_feature_list + [feature_id]
            candidate_accuracy = validator.evaluate(set(candidate_list), show_details=False) * 100.0
            print(f"Using feature(s) {format_feature_set(candidate_list)} accuracy is {candidate_accuracy:.1f}%")

            if candidate_accuracy > best_level_accuracy:
                best_level_accuracy = candidate_accuracy
                best_level_list = candidate_list

        if best_level_accuracy < best_overall_accuracy:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")


        if best_level_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_level_accuracy
            best_overall_list = best_level_list.copy()

        print(f"Feature set {format_feature_set(best_level_list)} was best, accuracy is {best_level_accuracy:.1f}%")
        current_feature_list = best_level_list

        if len(current_feature_list) == total_features:
            break

    return set(best_overall_list), best_overall_accuracy

def backward_elimination(total_features, validator):
    # start with ALL features, greedily remove one feature per level
    # at each level try dropping each current feature, keep best LOOCV accuracy
    current_feature_set = set(range(1, total_features + 1))
    base_accuracy = validator.evaluate(current_feature_set, show_details=False) * 100.0

    print(
        f'Running nearest neighbor with all {total_features} features, using "leaving-one-out" '
        f"evaluation, I get an accuracy of {base_accuracy:.1f}%"
    )
    print()
    print("Beginning search.")

    best_overall_set = current_feature_set.copy()
    best_overall_accuracy = base_accuracy

    for level_number in range(total_features - 1, 0, -1):
        print(f"\nEvaluating {get_level_text(level_number)}:")

        best_level_set = None
        best_level_accuracy = -1.0

        # nested loop: trying to remove each feature still in the set
        for feature_id in sorted(current_feature_set):
            candidate_set = current_feature_set - {feature_id}
            candidate_accuracy = validator.evaluate(candidate_set, show_details=False) * 100.0
            print(f"Using feature(s) {format_feature_set(candidate_set)} accuracy is {candidate_accuracy:.1f}%")

            if candidate_accuracy > best_level_accuracy:
                best_level_accuracy = candidate_accuracy
                best_level_set = candidate_set

        if best_level_accuracy < best_overall_accuracy:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        if best_level_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_level_accuracy
            best_overall_set = best_level_set.copy()

        print(f"Feature set {format_feature_set(best_level_set)} was best, accuracy is {best_level_accuracy:.1f}%")
        current_feature_set = best_level_set

        if len(current_feature_set) == 0:
            break

    return best_overall_set, best_overall_accuracy

# Formats a set of feature IDs as a string like '{1,3,5}'
def run_specific_feature_subset_test():
    print("\nTest Part 2 feature subsets:")
    print("    1) Small dataset - features {3, 5, 7}") # Hardcoded for the small set
    print("    2) Large dataset - features {1, 15, 27}") # Hardcoded for the large set
    try:
        selected_test = int(input().strip())
    except ValueError:
        print("Invalid input, please enter a 1 or 2.")
        return

    if selected_test == 1:
        file_name = "CS170_Small_DataSet__17.txt"
        selected_features = {3, 5, 7}
    elif selected_test == 2:
        file_name = "CS170_Large_DataSet__23.txt"
        selected_features = {1, 15, 27}
    else:
        print("Invalid choice")
        return

    print(f"\nReading data from {file_name}...")
    start_time = time.time()
    dataset_rows = load_data(file_name)
    load_duration = time.time() - start_time
    if len(dataset_rows) == 0:
        print("Error: No data loaded from file")
        return
    print(f"Time to load data: {load_duration:.4f} seconds")

    print("Normalizing features...")
    start_time = time.time()
    normalized_rows = normalize_features(dataset_rows)
    normalize_duration = time.time() - start_time
    print(f"Time to normalize features: {normalize_duration:.4f} seconds")
    print(f"Loaded {len(normalized_rows)} instances with {len(normalized_rows[0][1])} features")

    print("\nPerforming leave-one-out cross-validation...")
    classifier = NearestNeighbor()
    validator = Validator(classifier, normalized_rows)

    start_time = time.time()
    accuracy = validator.evaluate(selected_features, show_details=True)
    validation_duration = time.time() - start_time

    print(f"\nTime for leave-one-out validation: {validation_duration:.4f} seconds")
    print(f"Using feature(s) {format_feature_set(selected_features)}, accuracy is about {accuracy:.3f}")
    print(f"\nTotal time: {load_duration + normalize_duration + validation_duration:.4f} seconds")

def main():
   print("Welcome to Nikhil's and Akshay's Feature Selection Algorithm.")
   print("\nType in the name of the file to test: ", end="")
   file_name = input().strip()

   print("\nType the number of the algorithm you want to run.")
   print("    1) Forward Selection")
   print("    2) Backward Elimination")
   print("    3) Test Specific Feature Subset")
   try:
       selected_algorithm = int(input().strip())
   except ValueError:
       print("Invalid choice")
       return

   if selected_algorithm == 3:
       run_specific_feature_subset_test()
       return
   if selected_algorithm not in (1, 2):
       print("Invalid choice")
       return

   dataset_rows = load_data(file_name)
   if len(dataset_rows) == 0:
       print("Error: No data loaded from file")
       return

   total_features = len(dataset_rows[0][1])
   print(
       f"\nThis dataset has {total_features} features (not including the class attribute), "
       f"with {len(dataset_rows)} instances."
   )
   print()
   print("Please wait while I normalize the data... Done!")
   print()
   normalized_rows = normalize_features(dataset_rows)

   classifier = NearestNeighbor()
   validator = Validator(classifier, normalized_rows)

   if selected_algorithm == 1:
       best_features, best_accuracy = forward_selection(total_features, validator)
   elif selected_algorithm == 2:
       best_features, best_accuracy = backward_elimination(total_features, validator)
   print(
       f"\nFinished search!! The best feature subset is {format_feature_set(best_features)}, "
       f"which has an accuracy of {best_accuracy:.1f}%"
   )


if __name__ == "__main__":
   main()
import os
import time

# hardcoding datasets and and small localized helpers for future use
# load data, normalize features, score subsets with leave-one-out, run forward/backward search

# dataset picker — all files live in the datasets/ folder
DATASET_FOLDER = "datasets"
DATASETS = {
    1: ("Small dataset", "CS170_Small_DataSet__17.txt"),
    2: ("Large dataset", "CS170_Large_DataSet__23.txt"),
    # part 2 real-world dataset
    3: ("Real dataset", "wine_data.txt"),
}

# hardcoded feature subsets for the "test one subset" menu option
SUBSET_TESTS = {
    1: (1, {3, 5, 7}),
    2: (2, {1, 15, 27}),
}


def get_dataset_path(choice):
    # turn menu number into full file path like datasets/CS170_Small_DataSet__17.txt
    if choice not in DATASETS:
        return None
    return os.path.join(DATASET_FOLDER, DATASETS[choice][1])


def pick_dataset():
    # ask user to pick small, large, or real dataset (same numbered style as algorithm menu)
    print("\nWhich dataset do you want to use?")
    for number, (label, _) in DATASETS.items():
        print(f"    {number}) {label}")

    try:
        choice = int(input().strip())
    except ValueError:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return None

    file_path = get_dataset_path(choice)
    if file_path is None:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return None
    if not os.path.isfile(file_path):
        print(f"Error: could not find dataset file at {file_path}")
        return None

    label, file_name = DATASETS[choice]
    print(f"Selected {label} ({file_name})")
    return file_path

# data loading and preprocessing

def load_data(file_path):
    # load dataset from file, first column is class, rest are features
    # space-separated files while wine file uses commas instead
    dataset_rows = []

    with open(file_path, "r") as input_file:
        for raw_line in input_file:
            line_text = raw_line.strip()
            if not line_text:
                continue  # skip empty lines

            if "," in line_text:
                parts = [value.strip() for value in line_text.split(",") if value.strip()]
            else:
                parts = line_text.split()

            class_label = int(float(parts[0]))
            feature_values = [float(value) for value in parts[1:]]
            dataset_rows.append((class_label, feature_values))

    return dataset_rows

def normalize_features(dataset_rows):
    # min-max normalize each feature column: (value - min) / (max - min)
    # features in file aren't normalized, so we fix that here
    if not dataset_rows or not dataset_rows[0][1]:
        return dataset_rows

    total_features = len(dataset_rows[0][1])
    column_mins = [float("inf")] * total_features
    column_maxs = [float("-inf")] * total_features

    # go through all rows and find the smallest/largest value for each feature
    for class_label, feature_values in dataset_rows:
        for feature_index, value in enumerate(feature_values):
            if value < column_mins[feature_index]:
                column_mins[feature_index] = value
            elif value > column_maxs[feature_index]:
                column_maxs[feature_index] = value

    # now scale every value into the 0 to 1 range per column
    normalized_rows = []
    for class_label, feature_values in dataset_rows:
        normalized_values = []
        for feature_index, value in enumerate(feature_values):
            span = column_maxs[feature_index] - column_mins[feature_index]
            if span == 0:
                normalized_values.append(0.0)  # all same value in this column
            else:
                normalized_values.append((value - column_mins[feature_index]) / span)
        normalized_rows.append((class_label, normalized_values))

    return normalized_rows

def load_and_normalize(file_path):
    # one helper so we don't copy load + normalize steps in multiple places
    dataset_rows = load_data(file_path)
    if not dataset_rows:
        return None, None
    return dataset_rows, normalize_features(dataset_rows)

# nearest neighbor classifier

class NearestNeighbor:
    # simple 1-nearest-neighbor using euclidean distance
    # assignment says get this working before trying the search algorithms

    def __init__(self):
        self.training_rows = []  # list of (class_label, feature_values)

    def train(self, training_rows):
        # nn doesn't really train, it just remembers all the rows
        self.training_rows = training_rows

    def test(self, test_features, feature_subset):
        # guess the class for one test row using only the features in the subset
        # feature numbers are 1-indexed so they match assignment output like {1,3,5}
        if not self.training_rows:
            return None

        sorted_features = sorted(feature_subset)
        test_values = [test_features[feature_id - 1] for feature_id in sorted_features]

        best_distance = float("inf")
        best_label = None

        for train_label, train_features in self.training_rows:
            train_values = [train_features[feature_id - 1] for feature_id in sorted_features]

            # squared distance is enough, sqrt won't change which row is closest
            squared_sum = sum((left - right) ** 2 for left, right in zip(test_values, train_values))
            if squared_sum < best_distance:
                best_distance = squared_sum
                best_label = train_label

        return best_label
    
# validator wrapper (scores how good a feature subset is)

class Validator:
    # this is the "wrapper", & nn inside leave-one-out evaluation
    
    def __init__(self, classifier, dataset_rows):
        self.classifier = classifier
        self.dataset_rows = dataset_rows

    def evaluate(self, feature_subset, show_details=True):
        # leave-one-out cross validation:
        # hold out one row, train on the rest, see if we guess that row right
        # repeat for every row and count how many we got correct
        if not self.dataset_rows:
            return 0.0

        total_rows = len(self.dataset_rows)
        correct_predictions = 0

        for row_index in range(total_rows):
            # everything except the current row becomes training data
            training_rows = self.dataset_rows[:row_index] + self.dataset_rows[row_index + 1:]
            test_label, test_features = self.dataset_rows[row_index]

            self.classifier.train(training_rows)
            guessed_label = self.classifier.test(test_features, feature_subset)
            is_correct = guessed_label == test_label
            correct_predictions += is_correct

            # print one line per instance when testing a specific subset
            if show_details:
                print(
                    f"Instance Id: {row_index}, Correct Label: {float(test_label):.1f}, "
                    f"Guessed Label: {float(guessed_label):.1f}, Accurate: {is_correct}"
                )

        accuracy = correct_predictions / total_rows
        if show_details:
            print(f"\nCorrectly Classified {correct_predictions}/{total_rows} instances.")
            print(f"Accuracy: {accuracy:.2f}")

        return accuracy
    
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
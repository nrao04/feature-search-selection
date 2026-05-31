import math
import time

# data loading and preprocessing

def load_data(file_name):
    # load dataset from file
    # first column = class label, rest = feature values
    dataset_rows = []

    # open the file and read the data
    with open(file_name, "r") as input_file:
        for raw_line in input_file:
            split_values = raw_line.strip().split()
            # split the values by spaces
            if len(split_values) == 0:
                continue  # skip any blank lines

            class_label = int(float(split_values[0]))
            feature_values = [float(value) for value in split_values[1:]]
            dataset_rows.append((class_label, feature_values))

    return dataset_rows

def normalize_features(dataset_rows):
    # min-max normalize each feature column:
    # normalized = (value - min) / (max - min)
    if len(dataset_rows) == 0:
        return dataset_rows

    total_features = len(dataset_rows[0][1])
    if total_features == 0:
        return dataset_rows

    # initialize min and max trackers
    column_mins = [float("inf")] * total_features
    column_maxs = [float("-inf")] * total_features

    # scan all rows once to get each feature's min and max
    for class_label, feature_values in dataset_rows:
        for feature_index in range(total_features):
            current_value = feature_values[feature_index]
            if current_value < column_mins[feature_index]:
                column_mins[feature_index] = current_value
            if current_value > column_maxs[feature_index]:
                column_maxs[feature_index] = current_value

    # build normalized dataset
    normalized_rows = []
    for class_label, feature_values in dataset_rows:
        normalized_values = []
        for feature_index in range(total_features):
            min_value = column_mins[feature_index]
            max_value = column_maxs[feature_index]
            denominator = max_value - min_value

            if denominator == 0:
                # if all values in this column are the same, set to 0
                normalized_values.append(0.0)
            else:
                normalized_number = (feature_values[feature_index] - min_value) / denominator
                normalized_values.append(normalized_number)

        normalized_rows.append((class_label, normalized_values))

    return normalized_rows

class NearestNeighbor:
    # 1-nearest-neighbor: picking training row with smallest euclidean distance
    # assuming two classes and continuous features only

    def __init__(self):
        self.training_rows = []  # list of (class_label, feature_values)

    def train(self, training_rows):
        # NN just memorizes training data (no real "learning" step)
        self.training_rows = training_rows

    def test(self, test_features, feature_subset):
        # predict class for one test instance using only feature_subset columns
        # feature_subset is 1-indexed to match assignment / print format {1,2,3}
        if len(self.training_rows) == 0:
            return None

        sorted_features = sorted(feature_subset)
        test_subset_values = [test_features[feature_id - 1] for feature_id in sorted_features]

        best_distance = float("inf")
        best_label = None

        for train_label, train_features in self.training_rows:
            train_subset_values = [train_features[feature_id - 1] for feature_id in sorted_features]

            # euclidean distance on the selected features only
            squared_sum = 0.0
            for value_index in range(len(test_subset_values)):
                difference = test_subset_values[value_index] - train_subset_values[value_index]
                squared_sum += difference * difference

            distance = math.sqrt(squared_sum)
            if distance < best_distance:
                best_distance = distance
                best_label = train_label

        return best_label
    
class Validator:
    def __init__(self, classifier, dataset_rows):
        # classifier = our NearestNeighbor object
        # dataset_rows = full normalized dataset for this run
        self.classifier = classifier
        self.dataset_rows = dataset_rows

    def evaluate(self, feature_subset, show_details=True):
        # need a trace when testing a specific subset (show_details=True)
        # forward/backward search pass show_details=False so output isn't huge
        if len(self.dataset_rows) == 0:
            return 0.0

        total_rows = len(self.dataset_rows)
        correct_predictions = 0

        # leave-one-out: each instance gets to be the test row exactly once
        for row_index in range(total_rows):
            training_rows = self.dataset_rows[:row_index] + self.dataset_rows[row_index + 1:]
            test_label, test_features = self.dataset_rows[row_index]

            self.classifier.train(training_rows)
            guessed_label = self.classifier.test(test_features, feature_subset)

            prediction_is_correct = guessed_label == test_label
            if prediction_is_correct:
                correct_predictions += 1

            # per-instance trace lines (matches sample output in assignment)
            if show_details:
                print(
                    f"Instance Id: {row_index}, Correct Label: {float(test_label):.1f}, "
                    f"Guessed Label: {float(guessed_label):.1f}, Accurate: {prediction_is_correct}"
                )

        accuracy_decimal = correct_predictions / total_rows
        if show_details:
            print(f"\nCorrectly Classified {correct_predictions}/{total_rows} instances.")
            print(f"Accuracy: {accuracy_decimal:.2f}")

        return accuracy_decimal
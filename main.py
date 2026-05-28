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
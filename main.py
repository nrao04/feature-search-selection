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
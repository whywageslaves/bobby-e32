# VERSION Python 3.11.4
# See requirements.txt for required packages.
# visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for man, or
# https://scikit-learn.org/stable/modules/neighbors.html#regression for a complete guide.

import json
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# File path constants
MASTER_DATA_PATH = "master-data.json"
VALIDATION_DATA_PATH = "validation.json"
FINGERPRINTING_LOCATIONS_PATH = "demo/data/fingerprinting-locations.json"
VALIDATION_LOCATIONS_PATH = "demo/data/validation-locations.json"

def preprocessing(data, method='avg'):
    x = []
    y = []

    for entry in data:
        for key, values in entry.items():
            rssi_values = []
            for i in range(1, 8):  # Loop from 1 to 7 for bob1 to bob7
                beacon_key = f'bob{i}'
                if beacon_key in values:
                    # Convert the RSSI values to floats
                    rssi_list = [float(rssi) for rssi in values[beacon_key]]

                    if method == 'max':
                        # Replace value 42 with -200 and find the max
                        normalized_values = [-200 if rssi == 42 else rssi for rssi in rssi_list]
                        rssi_values.append(max(normalized_values))
                    else:  # method == 'avg'
                        # Exclude value 42 from the average calculation
                        filtered_values = [rssi for rssi in rssi_list if rssi != 42]
                        if filtered_values:
                            rssi_values.append(np.mean(filtered_values))
                        else:
                            rssi_values.append(np.nan)
                else:
                    rssi_values.append(np.nan)  # If key doesn't exist, append NaN

            x.append(rssi_values)
            y.append(key)

    return x, y

def load_training_test_data():
    with open(MASTER_DATA_PATH, "r") as file:
        train_json_data = json.load(file)
    X_train, y_train = preprocessing(train_json_data, 'max')
    with open(VALIDATION_DATA_PATH, "r") as file:
        test_json_data = json.load(file)
    X_test, y_test = preprocessing(test_json_data, 'max')

    return X_train, y_train, X_test, y_test

def load_ground_truth_data():
    with open(FINGERPRINTING_LOCATIONS_PATH, "r") as file:
        GT_train = json.load(file)[0]
    with open(VALIDATION_LOCATIONS_PATH, "r") as file:
        GT_test = json.load(file)[0]
    return GT_train, GT_test

def get_ground_truth_coords(GT_data, test_location_id):
    ground_truth_location = str(math.floor(float(test_location_id)))
    ground_truth_coords = GT_data[ground_truth_location]
    ground_truth_coords_arr = np.array([float(ground_truth_coords["x"]), float(ground_truth_coords["y"])])
    return ground_truth_coords_arr

def rename_label(y):
    group, direction_code = map(int, y.split('.'))
    directions = ["N", "E", "W", "S"]
    direction = directions[direction_code - 1]
    return f"{group}-{direction}"

def calculate_distance(test_coor, predicted_coor):
    return np.linalg.norm(test_coor - predicted_coor)

def calculate_metrics(df):
    # Extract the distances from the DataFrame and convert them to float
    distances = df['Distance (m)'].str.replace('m', '').astype(float)

    # Calculate metrics
    min_error = np.min(distances)
    max_error = np.max(distances)
    mean_error = np.mean(distances)
    median_error = np.median(distances)
    stddev_error = np.std(distances)
    percentile_25 = np.percentile(distances, 25)
    percentile_50 = np.percentile(distances, 50)
    percentile_75 = np.percentile(distances, 75)

    # Print the results
    print(f"Minimum error: {min_error:.2f}m")
    print(f"Maximum error: {max_error:.2f}m")
    print(f"Mean error: {mean_error:.2f}m")
    print(f"Median error: {median_error:.2f}m")
    print(f"Standard deviation: {stddev_error:.2f}m")
    print(f"25th percentile: {percentile_25:.2f}m")
    print(f"50th percentile: {percentile_50:.2f}m")
    print(f"75th percentile: {percentile_75:.2f}m")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_training_test_data()
    GT_train, GT_test = load_ground_truth_data()
    k = 1

    model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean")
    model.fit(X_train, y_train)

    distances, indices = model.kneighbors(X_test)

    results = []

    for i in range(len(X_test)):
        test_location = y_test[i]
        test_location_label = rename_label(test_location)
        test_coor = get_ground_truth_coords(GT_test, test_location)

        for j in range(k):
            predicted_location = y_train[indices[i][j]]
            predicted_location_label = rename_label(predicted_location)
            predicted_coor = get_ground_truth_coords(GT_train, predicted_location)
            distance = calculate_distance(test_coor, predicted_coor)

            # Append result as a dictionary
            results.append({
                'Test Location': test_location_label,
                'Closest Location': predicted_location_label,
                'Distance (m)': f'{distance:.2f}m'
            })

    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results)

    print(df)
    calculate_metrics(df)

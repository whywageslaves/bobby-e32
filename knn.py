import json
import math
import numpy as np
import pandas as pd
import itertools
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# File path constants
MASTER_DATA_PATH = "master-data.json"
VALIDATION_DATA_PATH = "validation.json"
FINGERPRINTING_LOCATIONS_PATH = "demo/data/fingerprinting-locations.json"
VALIDATION_LOCATIONS_PATH = "demo/data/validation-locations.json"


def preprocessing(data, beacons, method='avg'):
    x = []
    y = []

    for entry in data:
        for key, values in entry.items():
            rssi_values = []
            for beacon_key in beacons:
                if beacon_key in values:
                    rssi_list = [float(rssi) for rssi in values[beacon_key]]

                    if method == 'max':
                        normalized_values = [-200 if rssi == 42 else rssi for rssi in rssi_list]
                        rssi_values.append(max(normalized_values))
                    elif method == 'window':
                        normalized_values = [-200 if rssi == 42 else rssi for rssi in rssi_list]
                        # Take the max from windows of 5 and average 4 values
                        window_maxes = [max(normalized_values[i:i + 5]) for i in range(0, len(normalized_values), 5)]
                        rssi_values.append(np.mean(window_maxes[:4]))
                    else:  # method == 'avg'
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


def load_training_test_data(beacons, method):
    try:
        with open(MASTER_DATA_PATH, "r") as file:
            train_json_data = json.load(file)
        X_train, y_train = preprocessing(train_json_data, beacons, method)

        with open(VALIDATION_DATA_PATH, "r") as file:
            test_json_data = json.load(file)
        X_test, y_test = preprocessing(test_json_data, beacons, method)

        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


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
    directions = ["N", "E", "S", "W"]
    direction = directions[direction_code - 1]
    return f"{group}-{direction}"


def calculate_distance(test_coor, predicted_coor):
    return np.linalg.norm(test_coor - predicted_coor)


def calculate_metrics(df):
    distances = df['Distance (m)'].str.replace('m', '').astype(float)
    min_error = np.min(distances)
    max_error = np.max(distances)
    mean_error = np.mean(distances)
    median_error = np.median(distances)
    stddev_error = np.std(distances)
    percentile_25 = np.percentile(distances, 25)
    percentile_50 = np.percentile(distances, 50)
    percentile_75 = np.percentile(distances, 75)

    print(f"Minimum error: {min_error:.2f}m")
    print(f"Maximum error: {max_error:.2f}m")
    print(f"Mean error: {mean_error:.2f}m")
    print(f"Median error: {median_error:.2f}m")
    print(f"Standard deviation: {stddev_error:.2f}m")
    print(f"25th percentile: {percentile_25:.2f}m")
    print(f"50th percentile: {percentile_50:.2f}m")
    print(f"75th percentile: {percentile_75:.2f}m")


def evaluate_model(X_train, y_train, X_test, y_test, GT_train, GT_test, k=1):
    model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean")
    model.fit(X_train, y_train)
    distances, indices = model.kneighbors(X_test)

    results = []
    total_distance = 0.0

    for i in range(len(X_test)):
        test_location = y_test[i]
        test_location_label = rename_label(test_location)
        test_coor = get_ground_truth_coords(GT_test, test_location)
        cumulative_distance = 0.0

        for j in range(k):
            predicted_location = y_train[indices[i][j]]
            predicted_location_label = rename_label(predicted_location)
            predicted_coor = get_ground_truth_coords(GT_train, predicted_location)
            distance = calculate_distance(test_coor, predicted_coor)
            cumulative_distance += distance

            results.append({
                'Test Location': test_location_label,
                'Closest Location': predicted_location_label,
                'Distance (m)': f'{distance:.2f}m'
            })

        total_distance += cumulative_distance / k

    # Average the total distance over the number of test samples
    mean_error = total_distance / len(X_test)
    df = pd.DataFrame(results)

    return df, mean_error


def main():
    GT_train, GT_test = load_ground_truth_data()
    all_beacons = [f'bob{i}' for i in range(1, 8)]
    best_results = {'avg': {}, 'max': {}, 'window': {}}
    plotting_data = {'avg': [], 'max': [], 'window': []}
    k = 1

    for method in ['avg', 'max', 'window']:
    # for method in ['window']:
        for n in range(1, len(all_beacons) + 1):
            min_mean_error = float('inf')
            best_df = None
            best_beacons = None
            for beacons in itertools.combinations(all_beacons, n):
                X_train, y_train, X_test, y_test = load_training_test_data(beacons, method)
                if X_train and y_train and X_test and y_test:
                    df, mean_error = evaluate_model(X_train, y_train, X_test, y_test, GT_train, GT_test, k)
                    if mean_error < min_mean_error:
                        min_mean_error = mean_error
                        best_df = df
                        best_beacons = beacons
            best_results[method][n] = (min_mean_error, best_df, best_beacons)
            plotting_data[method].append(min_mean_error)
            print(f"\nMethod: {method}, Beacons: {n}, Best Beacons: {best_beacons}")
            print(best_df)
            calculate_metrics(best_df)

    # Plotting the results
    plt.plot(range(1, len(all_beacons) + 1), plotting_data['avg'], label='Average Preprocessing')
    plt.plot(range(1, len(all_beacons) + 1), plotting_data['max'], label='Max Preprocessing')
    plt.plot(range(1, len(all_beacons) + 1), plotting_data['window'], label='Window Preprocessing')
    plt.xlabel('Number of Beacons')
    plt.ylabel('Mean Error (m)')
    plt.title(f'Mean Error vs. Number of Beacons (K={k})')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

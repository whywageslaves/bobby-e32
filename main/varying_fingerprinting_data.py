from enum import Enum
import itertools
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class FingerprintingData(Enum):
    Full = 1,
    Half = 2,
    HalfLLM = 3

# Half of the fingerprinting data to keep, to ensure equal density.
HALF_IDS = set([1, 4, 6, 7, 9, 11, 15, 16, 18, 20, 22, 24, 25, 27, 29, 31])
LLM_INPUT_DATA_FILE = "dataset/llm_input_data2.json"
TRAINING_DATA = "dataset/training_data.json"

def parse_id(s):
    return int(s.split(".")[0])

with open(TRAINING_DATA, "r") as file:
    data = json.load(file)
# Process data to keep only the first two RSSI values for each bob
# for entry in data:
#     for id, bobs in entry.items():
#         for bob, values in bobs.items():
#             bobs[bob] = values[:2]
data = [{id: v for id, v in d.items() if parse_id(id) in HALF_IDS} for d in data]

with open(LLM_INPUT_DATA_FILE, 'w') as file:
    json.dump(data, file)

# print(data)

# class KNN:
#     # File path constants
#     TRAINING_DATA = "dataset/training_data.json"
#     LLM_TRAINING_DATA = "dataset/llm_training_data.json"
#     TRAINING_DATA_LOCATIONS = "dataset/training_data_locations.json"

#     VALIDATION_DATA = "dataset/validation_data.json"
#     VALIDATION_DATA_LOCATIONS = "dataset/validation_data_locations.json"

#     def __init__(self, fingerprinting_data=FingerprintingData.Full):
#         self.fingerprinting_data = fingerprinting_data

#     def _preprocessing(self, data, beacons, method='avg'):
#         x = []
#         y = []

#         for entry in data:
#             for key, values in entry.items():
#                 rssi_values = []
#                 for beacon_key in beacons:
#                     if beacon_key in values:
#                         rssi_list = [float(rssi) for rssi in values[beacon_key]]

#                         if method == 'max':
#                             normalized_values = [-200 if rssi == 42 else rssi for rssi in rssi_list]
#                             rssi_values.append(max(normalized_values))
#                         elif method == 'window':
#                             normalized_values = [-200 if rssi == 42 else rssi for rssi in rssi_list]
#                             # Take the max from windows of 5 and average 4 values
#                             window_maxes = [max(normalized_values[i:i + 5]) for i in
#                                             range(0, len(normalized_values), 5)]
#                             rssi_values.append(np.mean(window_maxes[:4]))
#                         else:  # method == 'avg'
#                             filtered_values = [rssi for rssi in rssi_list if rssi != 42]
#                             if filtered_values:
#                                 rssi_values.append(np.mean(filtered_values))
#                             else:
#                                 rssi_values.append(np.nan)
#                     else:
#                         rssi_values.append(np.nan)  # If key doesn't exist, append NaN

#                 x.append(rssi_values)
#                 y.append(key)

#         return x, y

#     def _load_training_test_data(self, beacons, method):
#         try:
#             with open(self.TRAINING_DATA, "r") as file:
#                 train_json_data = json.load(file)


#                 match self.fingerprinting_data:
#                     case FingerprintingData.Full:
#                         train_json_data = train_json_data

#                     case FingerprintingData.Half:
#                         train_json_data = [{id: v for id, v in d.items() if parse_id(id) in HALF_IDS} for d in train_json_data]

#                     case FingerprintingData.HalfLLM:
#                         with open(self.TRAINING_DATA, "r") as file:
#                             llm_train_json_data = json.load(file)
#                             llm_train_json_data= [{id: v for id, v in d.items() if parse_id(id) in HALF_IDS} for d in train_json_data]

#                         train_json_data = llm_train_json_data


#             X_train, y_train = self._preprocessing(train_json_data, beacons, method)

#             with open(self.VALIDATION_DATA, "r") as file:
#                 test_json_data = json.load(file)
#             X_test, y_test = self._preprocessing(test_json_data, beacons, method)

#             return X_train, y_train, X_test, y_test
#         except FileNotFoundError as e:
#             print(f"Error loading data: {e}")
#             return None, None, None, None

#     def _load_ground_truth_data(self, ):
#         with open(self.TRAINING_DATA_LOCATIONS, "r") as file:
#             GT_train = json.load(file)[0]
#             match self.fingerprinting_data:
#                 case FingerprintingData.Full:
#                     GT_train = GT_train

#                 case FingerprintingData.Half:
#                     GT_train = {id:xy for (id, xy) in GT_train.items() if int(id) in HALF_IDS}

#                 case FingerprintingData.HalfLLM:
#                     GT_train = {id:xy for (id, xy) in GT_train.items() if int(id) in HALF_IDS}

#             print(GT_train)
#         with open(self.VALIDATION_DATA_LOCATIONS, "r") as file:
#             GT_test = json.load(file)[0]
#         return GT_train, GT_test

#     def _get_ground_truth_coords(self, GT_data, test_location_id):
#         ground_truth_location = str(math.floor(float(test_location_id)))
#         ground_truth_coords = GT_data[ground_truth_location]
#         ground_truth_coords_arr = np.array([float(ground_truth_coords["x"]), float(ground_truth_coords["y"])])
#         return ground_truth_coords_arr

#     def _rename_label(self, y):
#         group, direction_code = map(int, y.split('.'))
#         directions = ["N", "E", "S", "W"]
#         direction = directions[direction_code - 1]
#         return f"{group}-{direction}"

#     def _calculate_distance(self, test_coor, predicted_coor):
#         return np.linalg.norm(test_coor - predicted_coor)

#     def _calculate_metrics(self, df):
#         distances = df['Distance (m)'].str.replace('m', '').astype(float)
#         min_error = np.min(distances)
#         max_error = np.max(distances)
#         mean_error = np.mean(distances)
#         median_error = np.median(distances)
#         stddev_error = np.std(distances)
#         percentile_25 = np.percentile(distances, 25)
#         percentile_50 = np.percentile(distances, 50)
#         percentile_75 = np.percentile(distances, 75)

#         print(f"Minimum error: {min_error:.2f}m")
#         print(f"Maximum error: {max_error:.2f}m")
#         print(f"Mean error: {mean_error:.2f}m")
#         print(f"Median error: {median_error:.2f}m")
#         print(f"Standard deviation: {stddev_error:.2f}m")
#         print(f"25th percentile: {percentile_25:.2f}m")
#         print(f"50th percentile: {percentile_50:.2f}m")
#         print(f"75th percentile: {percentile_75:.2f}m")

#     def _evaluate_model(self, X_train, y_train, X_test, y_test, GT_train, GT_test, k=1):
#         self.model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean")
#         self.model.fit(X_train, y_train)
#         distances, indices = self.model.kneighbors(X_test)

#         results = []
#         total_distance = 0.0

#         for i in range(len(X_test)):
#             test_location = y_test[i]
#             test_location_label = self._rename_label(test_location)
#             test_coor = self._get_ground_truth_coords(GT_test, test_location)
#             cumulative_distance = 0.0

#             for j in range(k):
#                 predicted_location = y_train[indices[i][j]]
#                 predicted_location_label = self._rename_label(predicted_location)
#                 predicted_coor = self._get_ground_truth_coords(GT_train, predicted_location)
#                 distance = self._calculate_distance(test_coor, predicted_coor)
#                 cumulative_distance += distance

#                 results.append({
#                     'Test Location': test_location_label,
#                     'Closest Location': predicted_location_label,
#                     'Distance (m)': f'{distance:.2f}m'
#                 })

#             total_distance += cumulative_distance / k

#         # Average the total distance over the number of test samples
#         mean_error = total_distance / len(X_test)
#         df = pd.DataFrame(results)

#         return df, mean_error

#     def _plot_result(self, all_beacons, k, plotting_data):
#         # Plotting the results
#         plt.plot(range(1, len(all_beacons) + 1), plotting_data['avg'], label='Average Preprocessing')
#         plt.plot(range(1, len(all_beacons) + 1), plotting_data['max'], label='Max Preprocessing')
#         plt.plot(range(1, len(all_beacons) + 1), plotting_data['window'], label='Window Preprocessing')
#         plt.xlabel('Number of Beacons')
#         plt.ylabel('Mean Error (m)')
#         plt.title(f'Mean Error vs. Number of Beacons (K={k})')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     def run_everything(self):
#         GT_train, GT_test = self._load_ground_truth_data()
#         all_beacons = [f'bob{i}' for i in range(1, 8)]
#         best_results = {'avg': {}, 'max': {}, 'window': {}}
#         plotting_data = {'avg': [], 'max': [], 'window': []}
#         k = 1

#         for method in ['avg', 'max', 'window']:
#             for n in range(1, len(all_beacons) + 1):
#                 min_mean_error = float('inf')
#                 best_df = None
#                 best_beacons = None
#                 for beacons in itertools.combinations(all_beacons, n):
#                     X_train, y_train, X_test, y_test = self._load_training_test_data(beacons, method)
#                     if X_train and y_train and X_test and y_test:
#                         df, mean_error = self._evaluate_model(X_train, y_train, X_test, y_test, GT_train, GT_test, k)
#                         if mean_error < min_mean_error:
#                             min_mean_error = mean_error
#                             best_df = df
#                             best_beacons = beacons
#                 best_results[method][n] = (min_mean_error, best_df, best_beacons)
#                 plotting_data[method].append(min_mean_error)
#                 print(f"\nMethod: {method}, Beacons: {n}, Best Beacons: {best_beacons}")
#                 print(best_df)
#                 self._calculate_metrics(best_df)

#         self._plot_result(all_beacons, k, plotting_data)

#     def run_for_demo(self, X_test_param):
#         GT_train, GT_test = self._load_ground_truth_data()
#         all_beacons = [f'bob{i}' for i in range(1, 8)]
#         k = 1
#         method = 'window'
#         X_train, y_train, X_test, y_test = self._load_training_test_data(all_beacons, method)
#         self._evaluate_model(X_train, y_train, X_test, y_test, GT_train, GT_test, k)

#         distances, indices = self.model.kneighbors(X_test_param)
#         # TODO: is this correct? I thought we wanted to take the average of the returned x and y locations?
#         return y_train[indices[0][0]]



# if __name__ == "__main__":
#     # main = KNN()
#     main = KNN(FingerprintingData.Half)
#     main.run_everything()

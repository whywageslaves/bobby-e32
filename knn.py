# VERSION Python 3.11.4
# See requirements.txt for required packages.
# visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for man, or
# https://scikit-learn.org/stable/modules/neighbors.html#regression for a complete guide.

from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from pprint import pprint

# Doesn't really do anything at the moment, just a wrapper for the sklearn function - will likely need to do
# some post-processing to get the actual location from the indices, which we will do here.

def knn(RSSI_train, RSSI_test, k): # location indices are implicit
    """
    Perform k-nearest neighbors classification.

    Parameters:
    RSSI_train (array-like): Training data features.
    RSSI_test (array-like): Test data features.
    k (int): Number of neighbors to consider.

    Returns:
    array-like: Predicted distances to the k nearest neighbors for the test data.
    array-like: Predicted labels for the test data.
    
    Sample training data
    RSSI_train = [
        [-60, -65, -70],  # RSSI values from 3 beacons for sample 1
        [-55, -60, -75],  # RSSI values from 3 beacons for sample 2
        # ...
    ]

    # LOCATION IS IMPLICIT
    LOCATION_train = [
        1,  # Location label for sample 1
        2,  # Location label for sample 2
        # ...
    ]

    # Sample test data
    RSSI_test = [
        [-58, -63, -68],  # RSSI values from 3 beacons for test sample 1
        [-53, -58, -73],  # RSSI values from 3 beacons for test sample 2
        # ...
    ]

    label_test = [
        1,  # Location label for test sample 1
        2,  # Location label for test sample 2
        # ...
    ]
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(RSSI_train)
    distances, indices = nbrs.kneighbors(RSSI_test)
    return distances, indices

def convert_from_json(json_data: dict):
    """
    Converts from the json format captured by the ESP-32 into the np arrays
    required for the knn algorithm.

    Input:
        json_data: JSON - Data to format

    Output:
        mapping: dict - Maps a returned index to a location_id
        np_array_data: np.array - Array of readings.
    """
    def combine_location_data(json_data: dict) -> dict:
        combined = {}

        for location_data in json_data:
            # location data is a dict containing
            # key: location_id, val: dict (key: beacon_ssid, val: list[rssi])
            (location_id, beacons_rssi) = list(location_data.items())[0]
            if location_id in combined:
                # Got multiple (separate) readings for one location.
                for beacon_ssid, rssi_values in beacons_rssi.items():
                    combined[location_id][beacon_ssid] += rssi_values
            else:
                combined[location_id] = beacons_rssi

        return combined
    
    combined = combine_location_data(json_data)
    # print(f"combined: {combined}")

    formatted = []
    mapping = {}

    # Calcuate average of beacon rssi values.
    for index, (location_id, beacons_rssi) in enumerate(combined.items()):
        # print(index, location_id, beacons_rssi)
        mapping[index] = location_id
        beacon_fmt_data = [(sum(int(x) for x in val) // len(val)) for (_, val) in sorted(beacons_rssi.items(), key=lambda item: item[1])]
        formatted.append(beacon_fmt_data)

    return (np.array(formatted), mapping)
        
if __name__ == "__main__":
    with open("sample-data-harry-p-fingerprinting-1.json", 'r') as file:
        json_data = json.load(file)
    
    (RSSI_train, mapping) = convert_from_json(json_data)
    # print(RSSI_train, mapping)

    # RSSI_train = np.array([[-60, -65, -70], [-55, -60, -75], [-50, -55, -80]])
    RSSI_test = np.array([[-58, -63, -68], [-53, -58, -73]])

    pprint(knn(RSSI_train, RSSI_test, 3))
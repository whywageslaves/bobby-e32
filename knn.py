# VERSION Python 3.11.4
# See requirements.txt for required packages.
# visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for man, or
# https://scikit-learn.org/stable/modules/neighbors.html#regression for a complete guide.

from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from pprint import pprint
import math

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
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean").fit(RSSI_train)
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
        # beacon_fmt_data = [(sum(int(x) for x in val) // len(val)) for (_, val) in sorted(beacons_rssi.items(), key=lambda item: item[0])]
        beacon_fmt_data = [max(int(x) for x in val) for (_, val) in sorted(beacons_rssi.items(), key=lambda item: item[0])]
        formatted.append(beacon_fmt_data)

    return (np.array(formatted), mapping)
        
if __name__ == "__main__":
    with open("master-data.json", "r") as file:
        train_json_data = json.load(file)

    (RSSI_train, mapping) = convert_from_json(train_json_data)
    # print(RSSI_train, mapping)
    # print(mapping) 

    with open("validation.json", "r") as file:
        test_json_data = json.load(file)

    (RSSI_test, test_mapping) = convert_from_json(test_json_data)
    # print(RSSI_test)
    # print(test_mapping)
    
    with open("demo/data/fingerprinting-locations.json", "r") as file:
        sample_location_data = json.load(file)[0]
        
    with open("demo/data/validation-locations.json", "r") as file:
        validation_location_data = json.load(file)[0]

    # ORIGIN = np.array([0.0, 0.0])

    # Stores the values of the caluculated euclid error distances.
    euclid_err_arr = []
    
    for i in range(len(RSSI_test)):
        test_location_id = test_mapping[i]
        test_location_data = RSSI_test[i]    

        (distances, indices) = knn(RSSI_train, np.array([test_location_data]), 1)

        # Find the estimated location id's that were generated via knn.
        estimated_location_ids = [mapping[idx] for idx in indices[0]]

        print(f"[{i}] Results for test location: {test_location_id} | {test_location_data}")
        print(f"Closest points are: {[mapping[idx] for idx in indices[0]]}")
        
        ground_truth_location = str(math.floor(float(test_location_id)))
        ground_truth_coords = validation_location_data[ground_truth_location]
        ground_truth_coords_arr = np.array([float(ground_truth_coords["x"]), float(ground_truth_coords["y"])])
        # ground_truth_coords_arr = ORIGIN
        print(f"Ground truth location: {ground_truth_coords} | {ground_truth_coords_arr}")
        
        for location_id in estimated_location_ids:
            # Floor the location_id.
            location = str(math.floor(float(location_id)))
            coords: dict = sample_location_data[location]
            coords_arr = np.array([float(coords["x"]), float(coords["y"])])
            print(f"location_id: [{location_id}], location: [{location}]")
            print(f"estimated_location_coordinates: {coords} | {coords_arr}")
            
            # Calculate error distance by getting euclidean distance
            # between the coordinates of the estimated location, and the
            # coordinates the data was actually sampled at.
            
            err_dist = np.linalg.norm(ground_truth_coords_arr - coords_arr)
            print(f"Euclidean distance from ground truth: {err_dist} meters.")
            
            euclid_err_arr.append(err_dist)

        print(f"Distances: {distances}")
        print()


    # Calculate results based off euclid_err_arr.
    print("------------Results----------------")
    N = len(RSSI_test)
    
    if len(euclid_err_arr) != N:
        print("Error in calcuations ....")
        exit(1)
        
    print(f"min error:      {np.min(euclid_err_arr)}")
    print(f"max error:      {np.max(euclid_err_arr)}")
    print(f"mean error:     {np.mean(euclid_err_arr)}")
    print(f"median error:   {np.median(euclid_err_arr)}")
    print(f"stddev:         {np.std(euclid_err_arr)}")
    print(f"25 percentile:  {np.percentile(euclid_err_arr, 25)}")
    print(f"50 percentile:  {np.percentile(euclid_err_arr, 50)}")
    print(f"75 percentile:  {np.percentile(euclid_err_arr, 75)}")

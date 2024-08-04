# VERSION Python 3.11.4
# See requirements.txt for required packages.
# visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for man, or
# https://scikit-learn.org/stable/modules/neighbors.html#regression for a complete guide.

from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from pprint import pprint
import math
import gaussian as gauss

# Doesn't really do anything at the moment, just a wrapper for the sklearn function - will likely need to do
# some post-processing to get the actual location from the indices, which we will do here.

def knn(RSSI_train, RSSI_test, k): # location indices are implicit
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean").fit(RSSI_train)
    distances, indices = nbrs.kneighbors(RSSI_test)
    return distances, indices

def convert_from_json(json_data, skip, method="max"):
    """
    Converts from the json format captured by the ESP-32 into the np arrays
    required for the knn algorithm.

    Input:
        json_data: JSON - Data to format

    Output:
        mapping: dict - Maps a returned index to a location_id
        np_array_data: np.array - Array of readings.
    """
    # The `json_data` is a list of dictionaries, where
    # key: location_id, val: dict of beacons to a list of rssi values.
    beacon_rssi_data = {}
    for location in json_data:
        beacon_rssi_data.update(location)
    
    formatted = []
    mapping = {}
    i_counter = 0
    for idx, (location_id, beacons_rssi) in enumerate(beacon_rssi_data.items()):
        if idx % skip != 0:
            continue
        # location_id = location[0]
        # beacons_rssi = location[1]
        mapping[len(formatted)] = location_id
        
        beacon_filter = [1, 2, 3, 4, 5, 6, 7]
        beacon_filter.sort()
        
        # Array of the formatted beacon rssi values for the current location_id.
        # Example: location_id = "1.4", [-20.4, -32.7, -34.4, ...]
        curr_location_beacon_fmt = []

        for beacon_id in beacon_filter:
            rssi_values = [int(x) for x in beacons_rssi[f"bob{beacon_id}"]]
            if method == "max":
                curr_location_beacon_fmt.append(np.max(rssi_values))
            elif method == "avg":
                normalized_rssi_values = [x for x in rssi_values if x != 42]
                curr_location_beacon_fmt.append(np.average(normalized_rssi_values))
            elif method == "win":
                normalized_rssi_values = [x for x in rssi_values if x != 42]
                window_maxes = [max(normalized_rssi_values[i:i + 5]) for i in range(0, len(normalized_rssi_values), 5)]
                curr_location_beacon_fmt.append(np.max(window_maxes))
        
        formatted.append(curr_location_beacon_fmt)                            

    return (formatted, mapping)
    
def pre_gausiann_data(beacons_filter: set, skip):
    # The `json_data` is a list of dictionaries, where
    # key: location_id, val: dict of beacons to a list of rssi values.
    with open("gaussian-master-data.json", "r") as file:
        json_data = json.load(file)
    
    with open("demo/data/fingerprinting-locations.json", "r") as file:
        sample_location_data = json.load(file)[0]
        
    beacon_rssi_data = {}
    for location in json_data:
        beacon_rssi_data.update(location)
    
    LOC_train = []
    RSSI_train = []
    
    OOB_VALUE = -200
    
    for idx, (location_id, beacons_rssi) in enumerate(beacon_rssi_data.items()):
        # beacon_filter = [1, 2, 3, 4, 5, 6, 7]
        # beacon_filter.sort()
        if idx % skip != 0:
            continue
        # Find coordinates of current location.
        location = str(math.floor(float(location_id)))
        coords: dict = sample_location_data[location]
        coords_arr = np.array([float(coords["x"]), float(coords["y"])])
        #print(f"location_id: [{location_id}], location: [{location}]")
        #print(f"estimated_location_coordinates: {coords} | {coords_arr}")
        
        temp_beacon_filter = list(beacons_filter)
        temp_beacon_filter.sort()
        
        filtered_beacons_rssi = []
        for beacon_id in temp_beacon_filter:
            rssi_values = [int(x) for x in beacons_rssi[f"bob{beacon_id}"]]
            filtered_beacons_rssi.append(rssi_values)

        # Use zip to generate a tuple of len(temp_beacon_filter), which represents
        # the rssi values of (bob1, bob2, bob3, ...)
        generated_samples = list(zip(*filtered_beacons_rssi))
        #print(len(generated_samples))
        
        for generated_sample in generated_samples:
            if OOB_VALUE in generated_sample:
                # Discard values that have OOB in any of the beacons.
                continue
            
            # Add corresponding entry in LOC_train for each generated sample
            # at that location.
            LOC_train.append(coords_arr)
            RSSI_train.append(generated_sample)
        
    return (np.array(LOC_train), np.array(RSSI_train))
    
if __name__ == "__main__":
    with open("master-data.json", "r") as file:
        train_json_data = json.load(file)

    with open("validation.json", "r") as file:
        test_json_data = json.load(file)
    n_locations_divider = 2
    (RSSI_train, mapping) = convert_from_json(train_json_data, 1)#n_locations_divider)
    (RSSI_test, test_mapping) = convert_from_json(test_json_data, 1)
    
    with open("demo/data/fingerprinting-locations.json", "r") as file:
        sample_location_data = json.load(file)[0]
        
        
    with open("demo/data/validation-locations.json", "r") as file:
        validation_location_data = json.load(file)[0]

    # Stores the values of the caluculated euclid error distances.
    euclid_err_arr = []
    
    K = 1
    """
    n_locations_divider = 2
    # lakjsdfljasd;lfkjasd
    beacons_filter = {1, 2, 3, 4, 5, 6, 7}
    (LOC_train, RSSI_train_processed) = pre_gausiann_data(beacons_filter, n_locations_divider)
    #(tmp_LOC_train, tmp_RSSI_train) = pre_gausiann_data(beacons_filter, 1)
    if len(LOC_train) != len(RSSI_train_processed):
        print("Error in pre_gaussian_data function.")
        exit(1)
    
    #[print(x) for x in LOC_train]
    #[print(x) for x in tmp_LOC_train]
    #print(len(LOC_train))
    #print(len(tmp_LOC_train))

    x, y, fake_RSSI = gauss.gaussian_gen_data(LOC_train, RSSI_train_processed, 5)
    
    #pprint(sample_location_data)
    
    LENGTH_RSSI_TRAIN = len(RSSI_train)
    id_counter = 100
    for i in range(len(fake_RSSI)):
        x_a = str(x[i])
        y_a = str(y[i])
        new_coord = {"x": x_a, "y": y_a}
        sample_location_data[str(id_counter)] = new_coord
        new_RSSI_val = []
        for beacon in fake_RSSI[i]:
            new_RSSI_val.append(np.max(beacon))
        RSSI_train.append(new_RSSI_val)
        
        mapping[LENGTH_RSSI_TRAIN + i] = str(id_counter)
        id_counter += 1

    #print(RSSI_train)
    #print(mapping)
    #print(RSSI_train)
    #print(len(RSSI_train))
    #print(LENGTH_RSSI_TRAIN)
    #print(len(fake_RSSI))
    #print(sample_location_data)"""
    RSSI_train = np.array(RSSI_train)
    #p1 = np.array([-63, -68, -60, -69, -73, -63, -68])
    #p2 = np.array([-64, -70, -68, -63, -71, -71, -73])
    #print(np.linalg.norm(p1 - p2))
    
    #for i in range(len(fake_RSSI)):
    #    print(f"x::{x[i]}, y::{y[i]}, fake_RSSI::{fake_RSSI[i]}")
    
    #[print(x) for x in fake_RSSI]
    #print(len(RSSI_train))
    #print(len(fake_RSSI))
    
    #print(RSSI_test[:15])
    #[[bob1, bob2, bob3, bob4, bob5, bob6, bob7], [bob1, bob2, bob3, bob4, bob5, bob6, bob7]]
    #[[bob1, bob1, ..., bob1], [bob2, bob2], ...]
    #predicted_locations = gauss.gaussian_predict_location(RSSI_train[:1000], LOC_train[:1000], RSSI_test[:1000])
    
    """
    euclid_err_arr = []
    for i in range(len(RSSI_test)):
        test_location_id = test_mapping[i]
        test_location_data = RSSI_test[i]
        ground_truth_location = str(math.floor(float(test_location_id)))
        ground_truth_coords = validation_location_data[ground_truth_location]
        ground_truth_coords_arr = np.array([float(ground_truth_coords["x"]), float(ground_truth_coords["y"])])
        k_err_dist = np.linalg.norm(ground_truth_coords_arr - predicted_locations)
        euclid_err_arr.append(k_err_dist)
    
    print(f"min error:      {np.min(euclid_err_arr)}")
    #[print(x) for x in predicted_locations]
    """
    #exit(1)
    
    for i in range(len(RSSI_test)):
        test_location_id = test_mapping[i]
        test_location_data = RSSI_test[i]

        (distances, indices) = knn(RSSI_train, np.array([test_location_data]), K)

        # Find the estimated location id's that were generated via knn.
        estimated_location_ids = [mapping[idx] for idx in indices[0]]

        print(f"[{i}] Results for test location: {test_location_id} | {test_location_data}")
        print(f"Closest points are: {estimated_location_ids}")
        
        ground_truth_location = str(math.floor(float(test_location_id)))
        ground_truth_coords = validation_location_data[ground_truth_location]
        ground_truth_coords_arr = np.array([float(ground_truth_coords["x"]), float(ground_truth_coords["y"])])
        print(f"Ground truth location: {ground_truth_coords} | {ground_truth_coords_arr}")
        
        k_estimate_coords_arr = []
        
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
            k_estimate_coords_arr.append(coords_arr)
            # euclid_err_arr.append(err_dist)
            print()

        print(f"Distances: {distances}")

        print("Calculate the average of the estimated k-nearest locations.")
        # Take average of k-nearest neighbours coords, and use it to 
        # calculate the error distance based off the ground truth coordinates.
        k_estimate_coords_x = np.average([coord[0] for coord in k_estimate_coords_arr])
        k_estimate_coords_y = np.average([coord[1] for coord in k_estimate_coords_arr])
        k_estimate_coords = np.array([k_estimate_coords_x, k_estimate_coords_y])
        k_err_dist = np.linalg.norm(ground_truth_coords_arr - k_estimate_coords)
        print(f"K average coords: {k_estimate_coords}")
        print(f"K error dist from ground truth: {k_err_dist}")
        
        euclid_err_arr.append(k_err_dist)
        print()
        print("--------------------------------------------------------------------")

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

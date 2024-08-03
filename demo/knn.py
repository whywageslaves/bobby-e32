# VERSION Python 3.11.4
# See requirements.txt for required packages.
# visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for man, or
# https://scikit-learn.org/stable/modules/neighbors.html#regression for a complete guide.

import time
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from pprint import pprint
import matplotlib

# matplotlib.use("WebAgg")
matplotlib.use("GTK3Agg")

# Doesn't really do anything at the moment, just a wrapper for the sklearn function - will likely need to do
# some post-processing to get the actual location from the indices, which we will do here.

FINGERPRINTING_DATA_FILE = "../master-data.json"
DEMO_DATA_FILE = "data/demo3-rssi.json"
FINGERPRINTING_LOCATIONS_FILE = "data/fingerprinting-locations.json"
# FINGERPRINTING_LOCATIONS_FILE = "data/validation-locations.json"

# Map the location id to the actual (x,y) location.
# locations = dict()


def knn(RSSI_train, RSSI_test, k):  # location indices are implicit
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


class Knn():
    def __init__(self):
        with open(FINGERPRINTING_DATA_FILE, "r") as file:
            train_json_data = json.load(file)

        (RSSI_train, mapping) = convert_from_json(train_json_data)
        self.RSSI_train = RSSI_train
        self.mapping = mapping

    def estimate_location(self, test_location_data):
        # # for i, test_location_data in enumerate(RSSI_test, start=1):
        (distances, indices) = knn(self.RSSI_train, np.array([test_location_data]), 3)
        # print(f"[{id}] Results for test location: {test_location_data}")
        # print(f"Closest points are: {[self.mapping[idx] for idx in indices[0]]}")
        # print(f"Distances: {distances}")
        # print()

        return self.mapping[indices[0][0]]


# Unused code:
# RSSI_train = np.array([[-60, -65, -70], [-55, -60, -75], [-50, -55, -80]])
# RSSI_test = np.array([[-58, -63, -68], [-53, -58, -73]])


FLOORPLAN_FILE = "images/floorplan.png"


def scale_factor():
    # This measurement is the wall from the top of the living room.
    px_x1 = 327.66045796308947
    px_x2 = 1257.6965872473393

    # This pixel distance in the image maps to this real distance (meters).
    px_distance = px_x2 - px_x1
    real_distance = 8.2

    return px_distance / real_distance


# Origin is top left corner of of floor plan (not image, but wall of apartment).
px_origin_x = 75.73193535787505
px_origin_y = 88.1788399570355

scale_x = scale_factor()
scale_y = scale_x


def real_to_pixel_coord(x_real, y_real):
    px_x = px_origin_x + (x_real * scale_x)
    px_y = px_origin_y + (y_real * scale_y)
    return (px_x, px_y)


def pixel_to_real_coord(px_x, px_y):
    x_real = (px_x - px_origin_x) / scale_x
    y_real = (px_y - px_origin_y) / scale_y
    return (x_real, y_real)


def print_coordinates_in_floorplan():
    # Load the image
    img = Image.open(FLOORPLAN_FILE)

    # Function to click and print coordinates
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print(f'x = {ix}, y = {iy}')

    # Plot image and set up click event
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def create_plot(real_coordinates):
    img = Image.open(FLOORPLAN_FILE)
    dpi = 200

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")

    # Add points.
    for (id, real_x, real_y) in real_coordinates:
        (px_x, px_y) = real_to_pixel_coord(real_x, real_y)
        ax.scatter(px_x, px_y, color="red")
        # Annotate the point
        ax.annotate(id,  # Text to display
                    (px_x, px_y),  # The point to annotate
                    textcoords="offset points",  # How to position the text
                    xytext=(0, 10),  # Distance from text to points (x,y)
                    ha='center')  # Horizontal alignment can be left, right or center

    plt.show()


class Plot():
    def __init__(self):
        # Turn on interactive mode.
        plt.ion()

        self.img = Image.open(FLOORPLAN_FILE)
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)
        self.ax.axis("off")

        # plt.show(block=True)
        # plt.show()

    def add_point(self, real_x, real_y, color="red"):
        (px_x, px_y) = real_to_pixel_coord(real_x, real_y)
        self.ax.scatter(px_x, px_y, color=color)
        # Annotate the point
        # self.ax.annotate(sample_id, (px_x, px_y), textcoords="offset points", xytext=(0, 10), ha='center')
        # Redraw the canvas.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class LocationMap():
    def __init__(self):
        with open(FINGERPRINTING_LOCATIONS_FILE, "r") as file:
            locations_data = json.load(file)

        # locations = []
        self.locations = dict()

        for obj in locations_data:
            for (location_id, coords) in obj.items():
                (real_x, real_y) = (float(coords["x"]), float(coords["y"]))
                self.locations[location_id] = (real_x, real_y)

                # locations.append((location_id, float(coords["x"]), float(coords["y"])))
                # try:
                #     locations.append((float(coords["x"]), float(coords["y"])))
                # except ValueError:
                #     # Ignore coords that can't be parsed to float.
                #     pass
        # print(locations)

        # create_plot([(3.25, 1.3), (0, 0), (8.2, 0)])
        # create_plot(locations)


DEMO_REAL_LOCATIONS_FILE = "data/demo_real_locations.json"


def collect_demo_locations():
    plt.ion()

    # Load the image
    img = Image.open(FLOORPLAN_FILE)

    real_locations = dict()
    location_id = 1

    scatter = None
    text_annotation = None

    # Function to click and print coordinates
    def onclick(event):
        nonlocal location_id, scatter, text_annotation

        (px_x, px_y) = event.xdata, event.ydata
        (real_x, real_y) = pixel_to_real_coord(px_x, px_y)

        print(f'x = {real_x}, y = {real_y}')
        real_locations[str(location_id)] = {"x": real_x, "y": real_y}

        # # Remove all old drawn points.
        # if scatter:
        #     scatter.remove()
        # if text_annotation:
        #     text_annotation.remove()

        # Draw point
        scatter = ax.scatter(px_x, px_y, color="red")
        # Annotate the point
        text_annotation = ax.annotate(location_id, (px_x, px_y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Redraw the canvas.
        fig.canvas.draw()
        fig.canvas.flush_events()

        location_id += 1

    # Plot image and set up click event
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    # Once plot is closed, write to json file.
    with open(DEMO_REAL_LOCATIONS_FILE, "w") as file:
        json.dump(real_locations, file, indent=4)


TIME_BETWEEN_SAMPLES = 1.46


def run_demo():
    knn = Knn()
    location_map = LocationMap()
    plot = Plot()

    with open(DEMO_DATA_FILE, "r") as file:
        test_json_data = json.load(file)

    (RSSI_test, test_mapping) = convert_from_json(test_json_data)

    with open(DEMO_REAL_LOCATIONS_FILE, "r") as file:
        real_locations = json.load(file)

    for sample_id, test_location_data in enumerate(RSSI_test, start=1):
        print(f"id: {sample_id}")

        # Plot estimated location.
        estimated_location_id_directional = knn.estimate_location(test_location_data)
        # Strip out direction.
        estimated_location_id = str(estimated_location_id_directional).split(".")[0]
        (real_x, real_y) = location_map.locations[estimated_location_id]
        print("estimated: ", (real_x, real_y))
        plot.add_point(real_x, real_y, color="red")

        # Plot actual ground truth location.
        real_location = real_locations[str(sample_id)]
        (actual_x, actual_y) = (real_location["x"], real_location["y"])
        print("ground truth: ", (actual_x, actual_y))
        plot.add_point(actual_x, actual_y, color="blue")

        time.sleep(TIME_BETWEEN_SAMPLES)

        # run_knn(test_location_data)
        # (distances, indices) = knn(RSSI_train, np.array([test_location_data]), 3)
        # print(f"[{id}] Results for test location: {test_location_data}")
        # print(f"Closest points are: {[mapping[idx] for idx in indices[0]]}")
        # print(f"Distances: {distances}")
        # print()

    # print_coordinates_in_floorplan()


def main():
    run_demo()

    # collect_demo_locations()

    pass


if __name__ == "__main__":
    main()

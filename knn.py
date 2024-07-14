import json

import numpy as np
from sklearn.neighbors import NearestNeighbors


def preprocessing(data):
    x = []
    y = []

    for entry in data:
        for key, values in entry.items():
            avg_bob1 = np.mean([float(i) for i in values['bob1']])
            avg_bob2 = np.mean([float(i) for i in values['bob2']])
            avg_bob3 = np.mean([float(i) for i in values['bob3']])
            x.append([avg_bob1, avg_bob2, avg_bob3])
            y.append(int(key))

    return x, y


def load_data():
    with open("sample-data-harry-p-fingerprinting-1.json", "r") as file:
        train_json_data = json.load(file)
    X_train, y_train = preprocessing(train_json_data)
    with open("sample-data-harry-p-validating-1.json", "r") as file:
        test_json_data = json.load(file)
    X_test, _ = preprocessing(test_json_data)

    return X_train, y_train, X_test


def rename_label(y):
    y = int(y)
    group = (y - 1) // 4 + 1
    direction = ["N", "E", "W", "S"][(y - 1) % 4]
    return f"{group}-{direction}"


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    k = 3

    model = NearestNeighbors(n_neighbors=k)
    model.fit(X_train, y_train)

    distances, indices = model.kneighbors(X_test)

    for i in range(len(X_test)):
        print(f'Test Location: {X_test[i]}')
        for j in range(k):
            print(f'Closest Location {j}: {X_train[indices[i][j]]} ({rename_label(indices[i][j])})')
        print()

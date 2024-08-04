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


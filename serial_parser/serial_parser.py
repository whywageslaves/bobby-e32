import serial
import re
import json
import time
from collections import defaultdict

# The port that the ESP-32 is connected to.
PORT = "COM5"

# Regex pattern to extract ssid and rssi data for a given beacon.
DATA_REGEX = r"ssid:(bob[123]);rssi:(-?\d{1,3})"

# Default number of samples to generate.
# Do note that this is the total number of samples, so should
# always be a multiple of the number of beacons (3).
DATA_SAMPLE_NUM = 3 * 2

# Use timestamp to ensure unique file.
timestamp = str(time.time())[:10]
# Create empty file here. Saves headache of edge cases later
# when trying to dump to json. Also add empty array to avoid edge
# case of invalid json parsing from empty file.
output_filename = f"sample-data-{timestamp}.json"
with open(output_filename, "w+") as file:
    json.dump([], file)

print(f"Data will be output to [{output_filename}]")

def get_beacon_rssi_data(n: int):
    global ser
    # Flush input buffer to clear stale entries.
    # NOTE: Pretty sure that's what it does based
    # off func name.
    ser.reset_input_buffer()

    # Return the data in a dict of
    # key: ssid, val: List[rssi]
    beacon_rssi_data = defaultdict(list)

    # NOTE: Due to the way I implemented scanning on the 
    # ESP-32, should get even spread of samples (assuming n is a multiple
    # of the beacon count.)
    for i in range(n):
        # Decode from binary, and format.
        line = ser.readline()
        fmt_line = line.decode("utf-8").strip()
        data = re.search(DATA_REGEX, fmt_line)
        if data:
            # Extract the groups from the regex match.
            ssid = data.group(1)
            rssi = data.group(2)
            beacon_rssi_data[ssid].append(rssi)
            print(f"{i}: ssid: [{ssid}], rssi: [{rssi}]")

    return dict(beacon_rssi_data)

# Open the serial port.
ser = serial.Serial(PORT, baudrate=115200)

while True:
    # Get user input for a location id.
    location_id = None
    while not location_id:
        usr_input = input("Enter location id, or q to quit: ")
        if usr_input == "q":
            location_id = "q"
            break
        try:
            location_id = int(usr_input)
        except:
            pass

    # Quit program.
    if location_id == "q":
        break

    # Generate DATA_SAMPLE_NUM number of rssi samples at the 
    # current location.     
    print(f"Location: [{location_id}] - Generating samples.")
    beacon_rssi_data = get_beacon_rssi_data(DATA_SAMPLE_NUM)
    print(f"Location: [{location_id}] - Finished sample collection.")
    print(f"Beacon data: {beacon_rssi_data}")

    # Save the data here as json - just in case smth goes wrong
    # data will still be saved. Not the most efficient however.
    # NOTE: Really annoying, but due to the truncation behaviour of 
    # opening a file in write (or not overwriting with append), 
    # need to open the file twice, once for input and output.
    with open(output_filename, "r") as infile:
        json_data = json.load(infile)
        # print(f"old json: {json_data}")
        json_data.append(beacon_rssi_data)
        # print(f"modified json: {json_data}")
        with open(output_filename, "w") as outfile:
            json.dump(json_data, outfile)

ser.close()

# Old code: 
# while line := ser.readline():
#     # Decode from binary, and format.
#     fmt_line = line.decode("utf-8").strip()
#     data = re.search(DATA_REGEX, fmt_line)
    
#     # print(f"line: {fmt_line}")
#     # print(f"data: {data}")
#     if not data:
#         continue

#     # Extract the groups from the regex match.
#     ssid = data.group(1)
#     rssi = data.group(2)
#     print(f"Found ssid: {ssid}, rssi: {rssi}")
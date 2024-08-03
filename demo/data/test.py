import json
import sys

# File paths
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Read the JSON data from the file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Swap the x and y values
for obj in data:
    for key in obj:
        # Temporarily store the value of x
        temp = obj[key]['x']
        # Swap the values
        obj[key]['x'] = obj[key]['y']
        obj[key]['y'] = temp

# Write the modified data back to a new file
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print("The x and y values have been swapped and saved to", output_file_path)

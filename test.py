import json

# Load the data from the JSON file
with open('master-data.json', 'r') as file:
    data = json.load(file)


    # Function to check if all keys have 20 samples
    def check_sample_counts(data):
        for entry in data:
            for main_key, sub_dict in entry.items():
                for sub_key, values in sub_dict.items():
                    if len(values) != 20:
                        print(f"{main_key} - {sub_key} does not have 20 samples. It has {len(values)} samples.")
                    # else:
                    #     print(f"{main_key} - {sub_key} has 20 samples.")


    # Call the function with the loaded data
    check_sample_counts(data)

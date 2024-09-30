import os
import json
import ray
import psutil

@ray.remote
def read_json_file(json_path):
    """Reads and returns data from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def combine_json_files_parallel(json_paths, master_json_path, overwrite=False):
    """Combines all bearing JSON files into a master JSON file using Ray for parallel processing."""
    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=True)
    num_cpus = min(num_cpus // 3, len(json_paths))  # Use one-third of the CPUs or number of JSON files
    ray.init(num_cpus=num_cpus)

    # Initialize a list to hold the combined data
    combined_data = []

    # Determine if the master JSON should be replaced or appended
    if overwrite or not os.path.exists(master_json_path):
        mode = 'replace'
    else:
        # If appending to an existing file, load the existing data
        with open(master_json_path, 'r') as f:
            combined_data = json.load(f)

    # Submit parallel tasks to read each JSON file
    json_tasks = [read_json_file.remote(json_path) for json_path in json_paths]

    # Gather all the results from the parallel tasks
    json_results = ray.get(json_tasks)

    # Combine the results
    for result in json_results:
        combined_data.extend(result)

    # Save the combined data to the master JSON file
    with open(master_json_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print("All JSON files combined successfully into the master JSON file!")
    return master_json_path

if __name__ == "__main__":
    # Path to the folder containing the individual bearing JSON files
    json_folder = '/home/dhaval/thewall/s3/data/processed/bronze'  # Adjust this path as necessary
    master_json_path = '/home/dhaval/thewall/s3/data/processed/silver/bearing_data_master.json'
    overwrite = True

    # Get all JSON file paths from the folder
    json_paths = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')]

    print(f"Found {len(json_paths)} JSON files to combine: {json_paths}")

    # Combine JSON files in parallel using Ray
    combine_json_files_parallel(json_paths, master_json_path, overwrite)

    # Shut down Ray
    ray.shutdown()

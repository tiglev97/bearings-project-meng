import os
import pandas as pd
import ray
import psutil
import time
import json

@ray.remote
def process_bearing_folder(bearing_folder, bronze_path, data_root_path):
    """Processes all CSV files in a specific bearing folder and stores data in a single JSON file."""
    # Get bearing name
    bearing_folder_name = bearing_folder.split('/')[-1]
    split_name = bearing_folder.split('_')[0]

    folder_json_path = f'{bronze_path}/bearing_data_{bearing_folder_name}.json'

    # Initialize a list to hold the combined data (main data + time series)
    combined_data_list = []

    folder_path = os.path.join(data_root_path, bearing_folder)
    for filename in os.listdir(folder_path):
        if filename.startswith('acc') and filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            channel_1 = df.iloc[:, -2].tolist()
            channel_2 = df.iloc[:, -1].tolist()

            # Create timestamp column
            df['timestamp'] = pd.to_datetime(
                df.iloc[:, 0].astype(str) + ':' +
                df.iloc[:, 1].astype(str) + ':' +
                df.iloc[:, 2].astype(str),
                format='%H:%M:%S',
                errors='coerce'
            )

            # Create an entry for main data with associated time series
            combined_data_entry = {
                'id': len(combined_data_list) + 1,  # Auto-increment the id
                'identifier': bearing_folder_name.capitalize(),
                'timestamp': str(df['timestamp'].iloc[0]),
                'split': split_name.capitalize(),
                'timeseries': {
                    'channel1': channel_1,
                    'channel2': channel_2
                }
            }
            combined_data_list.append(combined_data_entry)

    # Save combined data to a JSON file
    with open(folder_json_path, 'w') as json_file:
        json.dump(combined_data_list, json_file, indent=4)

    print(f"Processed bearing folder: {bearing_folder} and saved to {folder_json_path}")
    return folder_json_path

def get_list_of_all_folders(root_data_path):
    folder_names = []
    for dataset in ['Training_set', 'Validation_set', 'Testing_set']:
        dataset_path = os.path.join(root_data_path, dataset)
        if os.path.isdir(dataset_path):
            bearing_folders = [
                os.path.join(dataset, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))
            ]
            if len(bearing_folders) > 0:
                folder_names.extend(bearing_folders)
    return folder_names


if __name__ == "__main__":
    
    data_root_path = '/home/dhaval/thewall/s3/data/raw/FEMTOBearingDataSet'
    bronze_path = '/home/dhaval/thewall/s3/data/processed/bronze'

    # Get all bearing folders
    folder_names = get_list_of_all_folders(data_root_path)    

    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=True)
    num_cpus = min(num_cpus // 3, len(folder_names))  # Use half of the CPUs or number of folders
    print(f"Number of CPUs used: {num_cpus}")
    ray.init(num_cpus=num_cpus)

    # Time the folder processing
    start_time_processing_overall = time.time()

    # Tasks
    tasks = [process_bearing_folder.remote(folder, bronze_path, data_root_path) for folder in folder_names]
    ray.get(tasks)

    # Total time taken
    end_time_processing_initial = time.time()
    print(f"Total time taken: {(end_time_processing_initial - start_time_processing_overall) / 60:.2f} minutes")
    ray.shutdown()

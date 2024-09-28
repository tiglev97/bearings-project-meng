import pandas as pd
import json
import os
import re

def excel_to_jsonl(folder_path, jsonl_file_path):

    try:
        # List all relevant files (csv, xlsx, xml) starting with 'acc'

        data_files = [f for f in os.listdir(folder_path) 
                      if (f.startswith('acc') and (f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.xml')))]
    
        if not data_files:
            print("No relevant files found in the folder.")
            return

        # Check if the output JSONL file already exists
        if os.path.exists(jsonl_file_path):
            print(f"Error: The file {jsonl_file_path} already exists.")
            return
        

        if not os.path.exists(jsonl_file_path):
            os.makedirs(jsonl_file_path)  # Create the directory if it does not exist

        unique_id = 1  # Initialize a unique ID counter

        
        # Open the output JSONL file for writing
        with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            print(f"Converting {len(data_files)} files to JSONL...")

            for data_file in data_files:
                data_file_path = os.path.join(folder_path, data_file)

                # Determine file type and read data accordingly
                try:
                    if data_file.endswith('.csv'):
                        data = pd.read_csv(data_file_path)
                    elif data_file.endswith('.xlsx'):
                        data = pd.read_excel(data_file_path)
                    elif data_file.endswith('.xml'):
                        data = pd.read_xml(data_file_path)
                    else:
                        print(f"Unsupported file format for {data_file}. Skipping.")
                        continue
                except Exception as read_error:
                    print(f"Error reading {data_file}: {read_error}. Skipping.")
                    continue

                # Rename the columns to standardize(will be adapted to the actual column names based on user input)
                data.columns = ['Hour', 'Minute', 'Second', 'Millisec', 'Time Series x', 'Time Series y']

                # Ensure required columns are present
                required_columns = ['Hour', 'Minute', 'Second','Millisec', 'Time Series x', 'Time Series y']
                if not all(col in data.columns for col in required_columns):
                    print(f"File {data_file} is missing required columns. Skipping.")
                    continue

                # Group data by Hour, Minute, Second and aggregate Time Series x, y
                grouped_data = data.groupby(['Hour', 'Minute', 'Second']).agg({
                    'Time Series x': list, 
                    'Time Series y': list
                }).reset_index()


                # Extract bearing information from the folder name
                folder_name = os.path.basename(folder_path)
                bearing_search = re.findall(r'\d+_\d+', folder_name)
                bearing = bearing_search[0] if bearing_search else 'unknown'

                # Convert grouped data to JSONL format
                for _, row in grouped_data.iterrows():
                    record = {
                        "id": unique_id,
                        "bearing": bearing,
                        "channel x": row['Time Series x'],
                        "channel y": row['Time Series y']
                    }

                    # Write the record to JSONL
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')

                    unique_id += 1  # Increment the unique ID for the next record

        print(f"Success: All files have been converted to {jsonl_file_path}.")
    
    except Exception as e:
        print(f"Error: {e}")


import json
import pandas as pd
import re
import logging


class DataCheckPipeline:
    def __init__(self, data):
        self.data = data
        self.df = pd.DataFrame(data)

    def check_missing_values(self):
        missing = self.df.isnull().sum()
        logging.info("Missing Values Check:")
        logging.info(f"\n{missing}\n")

    def check_data_types(self):
        # Define the expected types for each field
        # TODO: DYNAMICALLY DEFINE EXPECTED TYPES BASED ON THE DATA
        expected_types = {
            'id': int,
            'identifier': str,
            'timestamp': str,
            'timepoint_index': int,
            'split': str,
            'timeseries': dict
        }

        # Iterate through each entry and check for type consistency
        for entry in self.data:
            for key, expected_type in expected_types.items():
                if not isinstance(entry[key], expected_type):
                    logging.error(f"Inconsistent type for '{key}' in id {entry['id']}: Expected {expected_type}, got {type(entry[key])}.")
                    raise TypeError(f"Inconsistent type for '{key}' in id {entry['id']}: Expected {expected_type}, got {type(entry[key])}.")

        logging.info("Data Consistency check completed.\n")

    def check_timeseries_lengths(self):
        for entry in self.data:
            channels = entry['timeseries']
            lengths = [len(v) for v in channels.values()]
            if len(set(lengths)) != 1:
                logging.warning(f"Inconsistent lengths in timeseries for id {entry['id']} - {lengths}")
        logging.info("Timeseries length check completed.\n")

    def check_timestamps(self):
        # Here you could implement more sophisticated checks, such as timestamp format validation
        timestamp_format = re.compile(r"\d{2}:\d{2}:\d{2}")  # HH:MM:SS format
        for entry in self.data:
            if not timestamp_format.match(entry['timestamp']):
                logging.warning(f"Invalid timestamp format for id {entry['id']} - {entry['timestamp']}")
        logging.info("Timestamp format check completed.\n")

    def check_duplicates(self):
        # Create a DataFrame without the 'timeseries' column
        df_without_timeseries = self.df.drop(columns=['timeseries'])

        # Check for duplicates
        duplicates = df_without_timeseries[df_without_timeseries.duplicated()]

        logging.info("Duplicate Rows Check (excluding 'timeseries'):")
        if not duplicates.empty:
            logging.info(f"\n{duplicates}\n")
        else:
            logging.info("No duplicate rows found.\n")

    def run_pipeline(self):
        self.check_missing_values()
        self.check_data_types()
        self.check_timeseries_lengths()
        self.check_timestamps()
        self.check_duplicates()


# Main Block
if __name__ == "__main__":

    # Set up logging
    # Setup logging configuration in main
    log_filename = "/home/dhaval/thewall/s3/logging/data_check_pipeline.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # 'a' for append mode if you prefer
    )

    logging.info("Starting the data check pipeline...")

    # read json file
    with open('/home/dhaval/thewall/s3/data/processed/silver/bearing_data_master.json') as file:
        data = json.load(file)
    # log success
    logging.info("Data loaded successfully.")

    # Initialize the pipeline
    pipeline = DataCheckPipeline(data)

    # Run the pipeline
    try:
        pipeline.run_pipeline()
        logging.info("Data check pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred during the pipeline execution: {e}")

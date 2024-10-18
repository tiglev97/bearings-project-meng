import numpy as np
import logging
import pandas as pd
import re



def data_checks(df):
    """
    Perform a series of data validation checks on the input dataset (DataFrame).
    
    Parameters:
    df (DataFrame): Input DataFrame to perform data checks.
    
    Returns:
    df (DataFrame): Updated DataFrame after checks (outliers removed).
    """
    # Step 1: Check for missing columns
    channel_columns = ['id', 'identifier', 'bearing', 'split', 'timestamp', 'channel_x', 'channel_y']
    missing_columns = [col for col in channel_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"The following specified columns are missing in the DataFrame: {missing_columns}")
        raise KeyError(f"The following specified columns are missing in the DataFrame: {missing_columns}")
    logging.info("Missing Columns check completed.\n")

    # Step 2: Check for missing values
    missing = df.isnull().sum()
    logging.info("Missing Values Check:")
    logging.info(f"\n{missing}\n")

    # Step 3: Check for data type consistency
    expected_types = {
        'id': int,
        'identifier': str,
        'timestamp': str,
        'timepoint_index': int,
        'split': str,
        'timeseries': dict
    }
    
    # Iterate through each row in the DataFrame and check types
    for index, row in df.iterrows():
        for key, expected_type in expected_types.items():
            if key in row and not isinstance(row[key], expected_type):
                logging.error(f"Inconsistent type for '{key}' in id {row['id']}: Expected {expected_type}, got {type(row[key])}.")
                raise TypeError(f"Inconsistent type for '{key}' in id {row['id']}: Expected {expected_type}, got {type(row[key])}.")
    logging.info("Data type consistency check completed.\n")

    # Step 4: Check for timeseries length consistency
    for index, row in df.iterrows():
        if 'timeseries' in row:
            channels = row['timeseries']
            lengths = [len(v) for v in channels.values()]
            if len(set(lengths)) != 1:
                logging.warning(f"Inconsistent lengths in timeseries for id {row['id']} - {lengths}")
    logging.info("Timeseries length check completed.\n")

    # Step 5: Check for timestamp format validity (HH:MM:SS format)
    timestamp_format = re.compile(r"\d{2}:\d{2}:\d{2}")
    for index, row in df.iterrows():
        timestamp_str = str(row.get('timestamp', ''))
        if 'timestamp' in row and not timestamp_format.match(row['timestamp']):
            logging.warning(f"Invalid timestamp format for id {row['id']} - {row['timestamp']}")
    logging.info("Timestamp format check completed.\n")

    
    # Step 6: Check and remove outliers (95% confidence interval)
    threshold = 3
    channels=['channel_x', 'channel_y']
    for column in channels:
        # Since the data is a list, compute z-scores for each time series
        def compute_z_scores(ts):
            ts = np.array(ts)
            return (ts - np.mean(ts)) / np.std(ts)
        
        # Compute z-scores and check for anomalies
        df[f'{column}_z_scores'] = df[column].apply(compute_z_scores)
        df[f'{column}_anomalies'] = df[f'{column}_z_scores'].apply(lambda z: np.any(np.abs(z) > threshold))
        
        # Log anomalies if detected
        if df[f'{column}_anomalies'].any():
            logging.error(f"Anomalies detected in column '{column}':")
            print(f"Anomalies detected in column '{column}':")
            logging.error(df[df[f'{column}_anomalies']])
            print(df[df[f'{column}_anomalies']])

        logging.info("Data Consistency check completed.\n")

    # Return the updated DataFrame
    return df


# class DataCheckPipeline:
#     def __init__(self, data):
#         self.data = data
#         self.df = pd.DataFrame(data)

#     def check_missing_columns(self):
#         channel_columns = [ 'id','identifier','bearing',
#                             'split','timestamp','millisec',
#                             'channel x','channel y']
#         missing_columns = [col for col in channel_columns if col not in self.df.columns]
#         if missing_columns:
#             logging.error(f"The following specified columns are missing in the DataFrame: {missing_columns}")
#             raise KeyError(f"The following specified columns are missing in the DataFrame: {missing_columns}")
#         logging.info("Missing Columns check completed.\n")

#     def check_missing_values(self):
#         missing = self.df.isnull().sum()
#         logging.info("Missing Values Check:")
#         logging.info(f"\n{missing}\n")
    
#     def check_data_types(self):
#         # Define the expected types for each field
#         # TODO: DYNAMICALLY DEFINE EXPECTED TYPES BASED ON THE DATA
#         expected_types = {
#             'id': int,
#             'identifier': str,
#             'timestamp': str,
#             'timepoint_index': int,
#             'split': str,
#             'timeseries': dict
#         }

#         # Iterate through each entry and check for type consistency
#         for entry in self.data:
#             for key, expected_type in expected_types.items():
#                 if not isinstance(entry[key], expected_type):
#                     logging.error(f"Inconsistent type for '{key}' in id {entry['id']}: Expected {expected_type}, got {type(entry[key])}.")
#                     raise TypeError(f"Inconsistent type for '{key}' in id {entry['id']}: Expected {expected_type}, got {type(entry[key])}.")

#     def check_timeseries_lengths(self):
#         for entry in self.data:
#             channels = entry['timeseries']
#             lengths = [len(v) for v in channels.values()]
#             if len(set(lengths)) != 1:
#                 logging.warning(f"Inconsistent lengths in timeseries for id {entry['id']} - {lengths}")
#         logging.info("Timeseries length check completed.\n")

#     def check_timestamps(self):
#         # Here you could implement more sophisticated checks, such as timestamp format validation
#         timestamp_format = re.compile(r"\d{2}:\d{2}:\d{2}")  # HH:MM:SS format
#         for entry in self.data:
#             if not timestamp_format.match(entry['timestamp']):
#                 logging.warning(f"Invalid timestamp format for id {entry['id']} - {entry['timestamp']}")
#         logging.info("Timestamp format check completed.\n")

#     def check_duplicates(self):
#         # Create a DataFrame without the 'timeseries' column
#         df_without_timeseries = self.df.drop(columns=['timeseries'])

#         # Check for duplicates
#         duplicates = df_without_timeseries[df_without_timeseries.duplicated()]

#         logging.info("Duplicate Rows Check (excluding 'timeseries'):")
#         if not duplicates.empty:
#             logging.info(f"\n{duplicates}\n")
#         else:
#             logging.info("No duplicate rows found.\n")

#     def check_outlier(self):
#         numerical_columns = self.df.select_dtypes(include=['number']).columns  # Select only numerical columns
#         logging.info("Outlier Removal (95% Interval) Check:")

#         outliers_removed = 0
#         for col in numerical_columns:
#             # Calculate the 95% confidence interval
#             lower_bound = self.df[col].quantile(0.025)
#             upper_bound = self.df[col].quantile(0.975)

#             # Count how many rows are outliers
#             outliers_in_column = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
#             outliers_removed += len(outliers_in_column)

#             # Log information about the outliers
#             if not outliers_in_column.empty:
#                 logging.info(f"Outliers detected in column '{col}':\n{outliers_in_column}\n")

#             # Remove the outliers from the DataFrame
#             self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

#         logging.info(f"Total outliers removed: {outliers_removed}")
#         logging.info("Outlier removal process completed.\n")

#     def run_pipeline(self):
#         self.check_missing_columns()
#         self.check_missing_values()
#         self.check_data_types()
#         self.check_timeseries_lengths()
#         self.check_timestamps()
#         self.check_duplicates()
#         self.check_outlier()


# # Main Block
# if __name__ == "__main__":

#     # Set up logging
#     # Setup logging configuration in main
#     log_filename = "bearings-project-meng/Document/logging/data_check_pipeline.log"
#     logging.basicConfig(
#         filename=log_filename,
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         filemode='w'  # 'a' for append mode if you prefer
#     )

#     logging.info("Starting the data check pipeline...")

#     # read json file
#     with open('/home/dhaval/thewall/s3/data/processed/silver/bearing_data_master.json') as file:
#         data = json.load(file)
#     # log success
#     logging.info("Data loaded successfully.")

#     # Initialize the pipeline
#     pipeline = DataCheckPipeline(data)

#     # Run the pipeline
#     try:
#         pipeline.run_pipeline()
#         logging.info("Data check pipeline completed successfully.")
#     except Exception as e:
#         logging.error(f"Error occurred during the pipeline execution: {e}")

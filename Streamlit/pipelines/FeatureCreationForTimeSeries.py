import numpy as np
import pandas as pd
import json
from scipy.fft import fft, ifft
from scipy.signal import hilbert, stft
from scipy.stats import skew, kurtosis
import pywt
import logging
import time
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    Normalizer,
)

from pipelines.DataChecks import data_checks
from Streamlit.pipelines.JsonlConverter import data_frame_to_jsonl


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataCleanPipeline:
    def __init__(self, data):
        self.df = (
            data.copy()
        )  # Ensure a copy is made to prevent altering the original data
        logging.info("Pipeline initialized with data.")

    def regulate_column_types(self):
        logging.info("Regulating column types.")

        # Define the desired types, excluding the list columns from being processed by astype()
        desired_types = {
            "id": "int64",
            "identifier": "object",
            "bearing": "object",
            "split": "object",
        }

        # Convert columns to the desired types using astype()
        for column, dtype in desired_types.items():
            if column in self.df.columns:
                if dtype == "datetime64[ns]":
                    # Convert 'timestamp' with specific format
                    logging.info(f"Converting column '{column}' to datetime format.")
                    self.df[column] = pd.to_datetime(
                        self.df[column], format="%H:%M:%S", errors="coerce"
                    )
                else:
                    # Convert other columns to the desired types
                    self.df[column] = self.df[column].astype(dtype, errors="ignore")
                logging.info(f"Column '{column}' converted to type '{dtype}'.")
            else:
                logging.warning(
                    f"Column '{column}' not found in DataFrame. Skipping type regulation."
                )

        logging.info("Column type regulation completed.")
        return self.df

    # Missing Value Imputation
    def preprocess_data(self, strategy, target_column):
        logging.info(f"Starting missing value imputation with strategy: {strategy}.")
        df_cleaned = self.df.copy()  # Initialize df_cleaned properly

        if not isinstance(target_column, list):
            target_column = [target_column]

        if not all(col in df_cleaned.columns for col in target_column):
            logging.error(f"Target column '{target_column}' not found in data.")
            raise ValueError(f"Target column '{target_column}' not found in data.")

        if strategy == "Drop Missing Values":
            df_cleaned = df_cleaned.dropna()
            logging.info(
                f"Dropped rows with missing values. Remaining rows: {df_cleaned.shape[0]}."
            )

        elif strategy == "Forward Fill":
            df_cleaned = df_cleaned.fillna(method="ffill")
            logging.info("Performed forward fill for missing values.")

        elif strategy == "Backward Fill":
            df_cleaned = df_cleaned.fillna(method="bfill")
            logging.info("Performed backward fill for missing values.")

        else:
            logging.error(f"Invalid strategy: {strategy}. Choose a valid strategy.")
            raise ValueError(
                f"Invalid strategy: {strategy}. Valid strategies are: 'Drop Missing Values', 'Mean Imputation', 'Median Imputation', 'Mode Imputation', 'Forward Fill', 'Backward Fill'."
            )

        self.df[target_column] = df_cleaned[target_column]
        self.df = df_cleaned
        logging.info("Missing value imputation completed.\n")
        return df_cleaned

    # Scaling the data (Normalization or Standardization) for nested entries
    def scale_data(self, scaling_method, target_column):
        logging.info(f"Starting scaling with method: {scaling_method}.")
        df_cleaned = self.df.copy()

        if not isinstance(target_column, list):
            target_column = [target_column]

        for column in target_column:
            logging.info(f"Scaling column: {column}.")

            # Define a function to apply scaling to each nested entry (list/array)
            def scale_entry(entry):
                entry = np.array(entry).reshape(
                    -1, 1
                )  # Reshape to 2D array for sklearn
                if scaling_method == "Standard Scaler":
                    scaler = StandardScaler()
                elif scaling_method == "Min-Max Scaler":
                    scaler = MinMaxScaler()
                elif scaling_method == "Normalizer":
                    scaler = Normalizer()
                else:
                    logging.error(f"Invalid scaling method: {scaling_method}.")
                    raise ValueError(
                        f"Invalid scaling method: {scaling_method}. Valid options are 'Standard Scaler', 'Min-Max Scaler', 'Normalizer'."
                    )

                scaled_entry = scaler.fit_transform(
                    entry
                ).flatten()  # Flatten back to 1D after scaling
                return (
                    scaled_entry.tolist()
                )  # Convert to list to ensure list format is maintained

            # Apply scaling to each entry in the target column
            df_cleaned[column] = df_cleaned[column].apply(scale_entry)
            logging.info(f"Scaling applied to column: {column} using {scaling_method}.")

        self.df = df_cleaned
        logging.info("Scaling completed.\n")
        return df_cleaned

    # Encoding categorical columns
    def encode_categorical_columns(self, target_column):
        logging.info("Starting encoding of categorical columns.")
        df_encoded = self.df.copy()
        exclude_column = ['timestamp','identifier','Millisec']
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_cols:
            # Skip columns that are in target_column
            if col in target_column or col in exclude_column:
                logging.info(f"Skipping encoding for column: {col}.")
                continue

            # Check if column entries are lists or arrays and flatten them
            if df_encoded[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: (
                        x[0]
                        if isinstance(x, (list, np.ndarray)) and len(x) > 0
                        else None
                    )
                )

            # Apply encoding based on the number of unique values
            if (
                len(df_encoded[col].unique()) > 2
            ):  # Apply one-hot encoding for multi-class columns
                df_encoded = pd.get_dummies(
                    df_encoded, columns=[col], prefix=col, drop_first=False
                )
                logging.info(f"One-hot encoding applied to column: {col}.")
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(
                    df_encoded[col].astype(str)
                )  # Ensure all entries are strings
                logging.info(f"Label encoding applied to column: {col}.")

        self.df = df_encoded
        logging.info("Categorical encoding completed.\n")
        return df_encoded

    # Check for missing values in the dataset
    def check_missing_columns(self):
        logging.info("Checking for missing columns.")
        missing_data = self.df.isnull().sum()
        missing_columns = missing_data[missing_data > 0]
        if not missing_columns.empty:
            logging.info(f"Missing values detected in columns:\n{missing_columns}")
        else:
            logging.info("No missing values detected.")
        return missing_columns

    # Outlier removal using z-scores
    def outlier_removal(self):
        logging.info("Starting outlier removal.")
        threshold = 3
        channels = ["channel_x", "channel_y"]
        for column in channels:

            def compute_z_scores(ts):
                ts = np.array(ts)
                return (ts - np.mean(ts)) / np.std(ts)

            self.df[f"{column}_z_scores"] = self.df[column].apply(compute_z_scores)
            self.df[f"{column}_anomalies"] = self.df[f"{column}_z_scores"].apply(
                lambda z: np.any(np.abs(z) > threshold)
            )

            if self.df[f"{column}_anomalies"].any():
                logging.error(f"Anomalies detected in column '{column}':")
                logging.error(self.df[self.df[f"{column}_anomalies"]])

            logging.info(f"Outlier removal completed for column: {column}.\n")

        return self.df

    # Running the full pipeline
    def run_pipeline(self, missing_value_strategy, scaling_method, target_column):
        logging.info("Starting data cleaning pipeline.")

        self.regulate_column_types()
        logging.info(f"Regulate column types.")

        missing_columns = self.check_missing_columns()
        if not missing_columns.empty:
            print("Missing columns before imputation:")
            print(missing_columns)

        logging.info(
            f"Handling missing values using strategy: {missing_value_strategy}."
        )
        self.preprocess_data(
            strategy=missing_value_strategy, target_column=target_column
        )

        logging.info(f"Scaling data using method: {scaling_method}.")
        self.scale_data(scaling_method=scaling_method, target_column=target_column)

        logging.info("Encoding categorical columns.")
        self.encode_categorical_columns(target_column=target_column)

        self.outlier_removal()
        logging.info("Outlier Removal completed.\n")

        self.df = self.df.set_index(
            "id", drop=True
        )  # Set 'id' as the index and drop the column
        logging.info("Data cleaning pipeline completed.\n")
        return self.df


### Feature Engineering Code
def time_domain_features(df, channel_columns=["channel_x", "channel_y"]):
    logging.info("Extracting time-domain features...")

    for channel in channel_columns:

        if df[channel].apply(lambda x: isinstance(x, list)).all():
            df[f"{channel}_mean"] = df[channel].apply(np.mean)
            df[f"{channel}_median"] = df[channel].apply(np.median)
            df[f"{channel}_std"] = df[channel].apply(np.std)
            df[f"{channel}_var"] = df[channel].apply(np.var)
            df[f"{channel}_skew"] = df[channel].apply(lambda x: skew(x))
            df[f"{channel}_kurtosis"] = df[channel].apply(lambda x: kurtosis(x))
            df[f"{channel}_rms"] = df[channel].apply(
                lambda x: np.sqrt(np.mean(np.square(x)))
            )
            df[f"{channel}_ptp"] = df[channel].apply(np.ptp)
            df[f"{channel}_crest_factor"] = df[channel].apply(
                lambda x: np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x)))
            )
            df[f"{channel}_energy"] = df[channel].apply(lambda x: np.sum(np.square(x)))

            def compute_entropy(ts):
                histogram, bin_edges = np.histogram(ts, bins=10, density=True)
                pdf = histogram / np.sum(histogram)
                pdf = pdf[pdf > 0]
                return -np.sum(pdf * np.log2(pdf))

            df[f"{channel}_entropy"] = df[channel].apply(compute_entropy)
        else:
            logging.error(
                f"Values in column '{channel}' are not in the expected list format."
            )
            raise ValueError(
                f"Values in column '{channel}' are not in the expected list format."
            )
    logging.info("Time-domain features extracted successfully.")
    return df

    # data_frame_to_jsonl(df, 'time_domain_features','Gold')
    # logging.info("Time-domain features saved to JSONL file at Silver output.")


def frequency_domain_features(df, channel_columns=["channel_x", "channel_y"]):
    logging.info("Extracting frequency-domain features...")

    for channel in channel_columns:

        if df[channel].apply(lambda x: isinstance(x, list)).all():
            # Compute the FFT for each signal in the column
            df[f"{channel}_fft"] = df[channel].apply(lambda x: np.fft.fft(x))

            # Since FFT results are complex, we need to handle them appropriately
            # Compute the magnitude and phase of the FFT
            df[f"{channel}_fft_magnitude"] = df[f"{channel}_fft"].apply(
                lambda x: np.abs(x)[: len(x) // 2]
            )
            df[f"{channel}_fft_phase"] = df[f"{channel}_fft"].apply(
                lambda x: np.angle(x)[: len(x) // 2]
            )
            # Compute the frequencies corresponding to the FFT values
            n = (
                df[channel].apply(len).iloc[0]
            )  
            # use milliseconds to find the sample rate 
            freq = np.fft.fftfreq(n, d=1 / 25600)[: n // 2]
            df[f"{channel}_fft_freq"] = [freq] * len(
                df
            )  # Assign the same freq array to all rows

            # Compute the power spectrum
            df[f"{channel}_power_spectrum"] = df[f"{channel}_fft_magnitude"].apply(
                lambda x: x**2
            )

            # Compute statistical features of the FFT magnitude
            df[f"{channel}_fft_mean"] = df[f"{channel}_fft_magnitude"].apply(np.mean)
            df[f"{channel}_fft_std"] = df[f"{channel}_fft_magnitude"].apply(np.std)
            df[f"{channel}_fft_max"] = df[f"{channel}_fft_magnitude"].apply(np.max)

            # Find the frequency corresponding to the maximum FFT magnitude
            # df[f'{channel}_fft_freq_max'] = df.apply(
            #     lambda row: row[f'{channel}_fft_freq'][np.argmax(row[f'{channel}_fft_magnitude'])], axis=1)

            # # Compute spectral centroid
            # df[f'{channel}_spectral_centroid'] = df.apply(
            #     lambda row: np.sum(row[f'{channel}_fft_freq'] * row[f'{channel}_fft_magnitude']) / np.sum(row[f'{channel}_fft_magnitude']), axis=1)

            # # Compute spectral bandwidth
            # df[f'{channel}_spectral_bandwidth'] = df.apply(
            #     lambda row: np.sqrt(np.sum(((row[f'{channel}_fft_freq'] - row[f'{channel}_spectral_centroid']) ** 2) * row[f'{channel}_fft_magnitude']) / np.sum(row[f'{channel}_fft_magnitude'])), axis=1)

            # Compute analytic signal
            df[f"{channel}_analytic_signal"] = df[channel].apply(lambda x: hilbert(x))
            df[f"{channel}_amplitude_envelope"] = df[
                f"{channel}_analytic_signal"
            ].apply(lambda x: np.abs(x))
            df[f"{channel}_phase_envelope"] = df[f"{channel}_analytic_signal"].apply(
                lambda x: np.angle(x)
            )

            # Compute log power spectrum
            df[f"{channel}_log_power_spectrum"] = df[f"{channel}_power_spectrum"].apply(
                lambda x: np.log(x + 1e-10)
            )

            # Compute cepstrum
            df[f"{channel}_cepstrum"] = df[f"{channel}_log_power_spectrum"].apply(
                lambda x: np.abs(ifft(x).real)
            )
            df[f"{channel}_cepstrum_mean"] = df[f"{channel}_cepstrum"].apply(np.mean)
            df[f"{channel}_cepstrum_std"] = df[f"{channel}_cepstrum"].apply(np.std)
            df[f"{channel}_cepstrum_max"] = df[f"{channel}_cepstrum"].apply(np.max)

            # Remove complex columns before serialization
            complex_columns = [f"{channel}_fft", f"{channel}_analytic_signal"]
            df.drop(columns=complex_columns, inplace=True)
        else:
            logging.error(
                f"Values in column '{channel}' are not in the expected list format."
            )
            raise ValueError(
                f"Values in column '{channel}' are not in the expected list format."
            )

    logging.info("Frequency-domain features extracted successfully.")

    # data_frame_to_jsonl(df, 'frequency_domain_features','Gold')
    # logging.info("Frequency-domain features saved to JSONL file at Silver output.")

    return df


def time_frequency_features(df, channel_columns=["channel_x", "channel_y"]):
    logging.info("Extracting time-frequency domain features...")

    def compute_stft(ts):
        f, t, Zxx = stft(ts, fs=25600)
        stft_magnitude = np.abs(Zxx)
        stft_frequency = f
        stft_time = t
        stft_mean = np.mean(stft_magnitude)
        stft_std = np.std(stft_magnitude)
        stft_max = np.max(stft_magnitude)
        return stft_mean, stft_std, stft_max, stft_magnitude, stft_frequency, stft_time

    def compute_wavelet(ts):
        scales = np.arange(1, 64)
        wavelet = "cmor1.0-1.5"
        coefficients, frequencies = pywt.cwt(ts, scales, wavelet)
        wavelet_magnitude = np.abs(coefficients)
        wavelet_mean = np.mean(wavelet_magnitude)
        wavelet_std = np.std(wavelet_magnitude)
        wavelet_max = np.max(wavelet_magnitude)
        return wavelet_mean, wavelet_std, wavelet_max, wavelet_magnitude

    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():

            stft_results = df[channel].apply(compute_stft)
            df[f"{channel}_stft_mean"] = stft_results.apply(lambda x: x[0])
            df[f"{channel}_stft_std"] = stft_results.apply(lambda x: x[1])
            df[f"{channel}_stft_max"] = stft_results.apply(lambda x: x[2])
            df[f"{channel}_stft_magnitude"] = stft_results.apply(lambda x: x[3].tolist())
            df[f"{channel}_stft_frequency"] = stft_results.apply(lambda x: x[4].tolist())
            df[f"{channel}_stft_time"] = stft_results.apply(lambda x: x[5].tolist())

            # wavelet_results = df[channel].apply(compute_wavelet)
            # df[f"{channel}_wavelet_mean"] = wavelet_results.apply(lambda x: x[0])
            # df[f"{channel}_wavelet_std"] = wavelet_results.apply(lambda x: x[1])
            # df[f"{channel}_wavelet_max"] = wavelet_results.apply(lambda x: x[2])
            # df[f"{channel}_wavelet_magnitude"] = wavelet_results.apply(lambda x: x[3].tolist())
        else:
            logging.error(
                f"Values in column '{channel}' are not in the expected list format."
            )
            raise ValueError(
                f"Values in column '{channel}' are not in the expected list format."
            )

    logging.info("Time-frequency domain features extracted successfully.")

    # data_frame_to_jsonl(df, 'time_frequency_features','Gold')
    # logging.info("Time-frequency domain features saved to JSONL file at Silver output.")
    return df
 

def extract_features(df, channel_columns=["channel_x", "channel_y"]):

    # Extract time-domain features
    cleaned_df = df.copy()
    time_domain_feature_df = time_domain_features(cleaned_df, channel_columns)

    # Extract frequency-domain features
    cleaned_df=df.copy()
    frequency_domain_features_df = frequency_domain_features(cleaned_df, channel_columns)

    # Extract time-frequency domain features
    cleaned_df = df.copy()
    time_frequency_features_df = time_frequency_features(cleaned_df, channel_columns)

    # return time_domain_feature_df

    return time_domain_feature_df, frequency_domain_features_df,time_frequency_features_df


# # Usage example
# start_time = time.time()


# file_path = 'C:/uoft/Meng_project/bearings-project-meng/Streamlit/outputs/Bearing1_1.jsonl'
# df = load_jsonl_to_dataframe(file_path)
# print(df.head())
# time_domain_features_df, frequency_domain_features_df, time_frequency_features_df = extract_features(df)
# # Displaying the first few rows of the DataFrame with new features
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# logging.info(f"Time taken: {end_time - start_time} seconds")

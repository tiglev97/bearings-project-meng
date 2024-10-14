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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, Normalizer

from pipelines.DataChecks import data_checks

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleanPipeline:
    def __init__(self, data):
        self.data = data
        self.df = pd.DataFrame(data)
        logging.info("Pipeline initialized with data.")

    # Missing Value Imputation
    def preprocess_data(self, strategy, target_column):
        logging.info(f"Starting missing value imputation with strategy: {strategy}.")
        df_cleaned = self.df.copy()  # Start with a copy of the original DataFrame

        # Exclude the target column from processing
        df_without_target = df_cleaned.drop(columns=[target_column])

        if strategy == 'Drop Missing Values':
            df_cleaned = df_cleaned.dropna()
            logging.info(f"Dropped rows with missing values. Remaining rows: {df_cleaned.shape[0]}.")
            
        elif strategy in ['Mean Imputation', 'Median Imputation']:
            # Check for numeric columns only
            numeric_cols = df_without_target.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy=strategy.split()[0].lower())
                df_cleaned[numeric_cols] = imputer.fit_transform(df_without_target[numeric_cols])
                logging.info(f"Performed {strategy.lower()} for numeric columns.")
            
            # Impute for categorical columns
            categorical_cols = df_without_target.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_cleaned[categorical_cols] = imputer.fit_transform(df_without_target[categorical_cols])
                logging.info("Performed most frequent imputation for categorical columns.")

        elif strategy == 'Mode Imputation':
            # Apply mode imputation to categorical columns
            categorical_cols = df_without_target.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_cleaned[categorical_cols] = imputer.fit_transform(df_without_target[categorical_cols])
                logging.info("Performed mode imputation for categorical columns.")

        elif strategy == 'Forward Fill':
            df_cleaned = df_cleaned.fillna(method='ffill')
            logging.info("Performed forward fill for missing values.")

        elif strategy == 'Backward Fill':
            df_cleaned = df_cleaned.fillna(method='bfill')
            logging.info("Performed backward fill for missing values.")

        self.df = df_cleaned
        logging.info("Missing value imputation completed.\n")
        return df_cleaned

    # Scaling the data (Normalization or Standardization)
    def scale_data(self, scaling_method, target_column):
        logging.info(f"Starting scaling with method: {scaling_method}.")
        df_cleaned = self.df.copy()  # Ensure a fresh copy for scaling
        
        # Select numerical columns
        numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        
        # Exclude binary, one-hot encoded, and target columns
        binary_cols = df_cleaned.columns[df_cleaned.nunique() <= 2].tolist()  # Binary columns
        numeric_cols = [col for col in numeric_cols if col not in binary_cols and col != target_column]

        if not numeric_cols:
            warnings.warn("No numerical columns to scale.")
            logging.info("No numerical columns available for scaling.")
            return df_cleaned  # Return the original df if no suitable numerical columns

        # Apply the selected scaling method
        if scaling_method == 'Standard Scaler':
            scaler = StandardScaler()
            df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
            logging.info("Standard scaling applied to numerical columns.")
        elif scaling_method == 'Min-Max Scaler':
            scaler = MinMaxScaler()
            df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
            logging.info("Min-Max scaling applied to numerical columns.")
        elif scaling_method == 'Normalizer':
            scaler = Normalizer()
            df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
            logging.info("Normalizer applied to numerical columns.")
    
        self.df = df_cleaned
        logging.info("Scaling completed.\n")
        return df_cleaned
    
    # Encoding categorical columns
    def encode_categorical_columns(self, target_column):
        logging.info(f"Starting encoding of categorical columns, target column: {target_column}.")
        df_encoded = self.df.copy()  # Avoid modifying the original

        # Identify categorical columns, excluding the target column
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col == target_column:
                # Label encoding for the target column
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                logging.info(f"Label encoding applied to target column: {col}.")
            else:
                if len(df_encoded[col].unique()) > 2:  # More than 2 unique values
                    # One-Hot Encoding
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=True)
                    logging.info(f"One-hot encoding applied to column: {col}.")
                else:
                    # Label encoding for binary categorical columns
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    logging.info(f"Label encoding applied to column: {col}.")
        
        self.df = df_encoded
        logging.info("Categorical encoding completed.\n")
        return df_encoded

    # Check for missing values in the dataset
    def check_missing_columns(self):
        logging.info("Checking for missing columns.")
        missing_data = self.df.isnull().sum()
        missing_columns = missing_data[missing_data > 0]  # Return columns with missing data
        if not missing_columns.empty:
            logging.info(f"Missing values detected in columns:\n{missing_columns}")
        else:
            logging.info("No missing values detected.")
        return missing_columns

    # Running the full pipeline
    def run_pipeline(self, missing_value_strategy, scaling_method, target_column):
        # Step 1: Check for missing values
        logging.info("Starting data cleaning pipeline.")
        missing_columns = self.check_missing_columns()
        if not missing_columns.empty:
            print("Missing columns before imputation:")
            print(missing_columns)
        
        # Step 2: Handle missing values based on the chosen strategy
        logging.info(f"Handling missing values using strategy: {missing_value_strategy}.")
        self.preprocess_data(strategy=missing_value_strategy, target_column=target_column)

        # Step 3: Scale the numerical data
        logging.info(f"Scaling data using method: {scaling_method}.")
        self.scale_data(scaling_method=scaling_method, target_column=target_column)

        # Step 4: Encode categorical columns
        logging.info("Encoding categorical columns.")
        self.encode_categorical_columns(target_column=target_column)

        logging.info("Data cleaning pipeline completed.\n")
        # Final DataFrame after the pipeline is run
        return self.df

        

def load_jsonl_to_dataframe(file_path):
    logging.info(f"Loading JSONL file from {file_path}...")
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            if 'time_series' in record:
                time_series_data = record.pop('time_series')
                record.update(time_series_data)
            data.append(record)
    df = pd.DataFrame(data)
    
    logging.info("Available columns in DataFrame: %s", df.columns)
    return df

def time_domain_features(df, channel_columns=['channel_x', 'channel_y']):
    logging.info("Extracting time-domain features...")
    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():
            df[f'{channel}_mean'] = df[channel].apply(np.mean)
            df[f'{channel}_median'] = df[channel].apply(np.median)
            df[f'{channel}_std'] = df[channel].apply(np.std)
            df[f'{channel}_var'] = df[channel].apply(np.var)
            df[f'{channel}_skew'] = df[channel].apply(lambda x: skew(x))
            df[f'{channel}_kurtosis'] = df[channel].apply(lambda x: kurtosis(x))
            df[f'{channel}_rms'] = df[channel].apply(lambda x: np.sqrt(np.mean(np.square(x))))
            df[f'{channel}_ptp'] = df[channel].apply(np.ptp)
            df[f'{channel}_crest_factor'] = df[channel].apply(lambda x: np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x))))
            df[f'{channel}_energy'] = df[channel].apply(lambda x: np.sum(np.square(x)))
            def compute_entropy(ts):
                histogram, bin_edges = np.histogram(ts, bins=10, density=True)
                pdf = histogram / np.sum(histogram)
                pdf = pdf[pdf > 0]
                return -np.sum(pdf * np.log2(pdf))
            df[f'{channel}_entropy'] = df[channel].apply(compute_entropy)
        else:
            logging.error(f"Values in column '{channel}' are not in the expected list format.")
            raise ValueError(f"Values in column '{channel}' are not in the expected list format.")
    logging.info("Time-domain features extracted successfully.")
    return df

def frequency_domain_features(df, channel_columns=['channel_x', 'channel_y']):
    logging.info("Extracting frequency-domain features...")
    def compute_fft(channel_data):
        fft_vals = np.fft.fft(channel_data)
        fft_freqs = np.fft.fftfreq(len(fft_vals))
        amplitude = np.abs(fft_vals)
        return fft_freqs[np.argmax(amplitude)]

    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():
            df[f'{channel}_fft'] = df[channel].apply(fft)
            df[f'{channel}_fft_magnitude'] = df[f'{channel}_fft'].apply(lambda x: np.abs(x)[:len(x)//2])
            df[f'{channel}_fft_freq'] = df[f'{channel}_fft_magnitude'].apply(lambda x: np.fft.fftfreq(len(x)*2, d=1)[:len(x)])
            df[f'{channel}_power_spectrum'] = df[f'{channel}_fft_magnitude'].apply(lambda x: x ** 2)
            df[f'{channel}_fft_mean'] = df[f'{channel}_fft_magnitude'].apply(np.mean)
            df[f'{channel}_fft_std'] = df[f'{channel}_fft_magnitude'].apply(np.std)
            df[f'{channel}_fft_max'] = df[f'{channel}_fft_magnitude'].apply(np.max)
            df[f'{channel}_fft_freq_max'] = df[f'{channel}_fft'].apply(lambda x: np.argmax(np.abs(x)))

            def spectral_centroid(magnitude, freq):
                return np.sum(freq * magnitude) / np.sum(magnitude)
            df[f'{channel}_spectral_centroid'] = df.apply(lambda row: spectral_centroid(row[f'{channel}_fft_magnitude'], row[f'{channel}_fft_freq']), axis=1)

            def spectral_bandwidth(magnitude, freq, centroid):
                return np.sqrt(np.sum(((freq - centroid) ** 2) * magnitude) / np.sum(magnitude))
            df[f'{channel}_spectral_bandwidth'] = df.apply(lambda row: spectral_bandwidth(row[f'{channel}_fft_magnitude'], row[f'{channel}_fft_freq'], row[f'{channel}_spectral_centroid']), axis=1)

            df[f'{channel}_analytic_signal'] = df[channel].apply(hilbert)
            df[f'{channel}_amplitude_envelope'] = df[f'{channel}_analytic_signal'].apply(np.abs)
            df[f'{channel}_phase_envelope'] = df[f'{channel}_analytic_signal'].apply(np.angle)

            df[f'{channel}_log_power_spectrum'] = df[f'{channel}_power_spectrum'].apply(lambda x: np.log(x + 1e-10))
            df[f'{channel}_cepstrum'] = df[f'{channel}_log_power_spectrum'].apply(lambda x: np.abs(ifft(x).real))
            df[f'{channel}_cepstrum_mean'] = df[f'{channel}_cepstrum'].apply(np.mean)
            df[f'{channel}_cepstrum_std'] = df[f'{channel}_cepstrum'].apply(np.std)
            df[f'{channel}_cepstrum_max'] = df[f'{channel}_cepstrum'].apply(np.max)
        else:
            logging.error(f"Values in column '{channel}' are not in the expected list format.")
            raise ValueError(f"Values in column '{channel}' are not in the expected list format.")
        
    logging.info("Frequency-domain features extracted successfully.")
    return df

def time_frequency_features(df, channel_columns=['channel_x', 'channel_y']):
    logging.info("Extracting time-frequency domain features...")
    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():
            def compute_stft(ts):
                f, t, Zxx = stft(ts, nperseg=128)
                stft_magnitude = np.abs(Zxx)
                stft_mean = np.mean(stft_magnitude)
                stft_std = np.std(stft_magnitude)
                stft_max = np.max(stft_magnitude)
                return stft_mean, stft_std, stft_max
            stft_results = df[channel].apply(compute_stft)
            df[f'{channel}_stft_mean'] = stft_results.apply(lambda x: x[0])
            df[f'{channel}_stft_std'] = stft_results.apply(lambda x: x[1])
            df[f'{channel}_stft_max'] = stft_results.apply(lambda x: x[2])

            def compute_wavelet(ts):
                scales = np.arange(1, 128)
                wavelet = 'cmor1.0-1.5'
                coefficients, frequencies = pywt.cwt(ts, scales, wavelet)
                wavelet_magnitude = np.abs(coefficients)
                wavelet_mean = np.mean(wavelet_magnitude)
                wavelet_std = np.std(wavelet_magnitude)
                wavelet_max = np.max(wavelet_magnitude)
                return wavelet_mean, wavelet_std, wavelet_max
            wavelet_results = df[channel].apply(compute_wavelet)
            df[f'{channel}_wavelet_mean'] = wavelet_results.apply(lambda x: x[0])
            df[f'{channel}_wavelet_std'] = wavelet_results.apply(lambda x: x[1])
            df[f'{channel}_wavelet_max'] = wavelet_results.apply(lambda x: x[2])
        else:
            logging.error(f"Values in column '{channel}' are not in the expected list format.")
            raise ValueError(f"Values in column '{channel}' are not in the expected list format.")
        
    logging.info("Time-frequency domain features extracted successfully.")
    return df

# def extract_features(df, channel_columns=['channel_x', 'channel_y']):
#     logging.info("Starting feature extraction process...")
#     time_df = time_domain_features(df, channel_columns)
#     frequencey_df = frequency_domain_features(df, channel_columns)
#     time_frequencey_df = time_frequency_features(df, channel_columns)
#     logging.info("Feature extraction completed.")
#     return 
# Combine all feature extraction functions




def extract_features(df, channel_columns=['channel_x', 'channel_y']):
    # Perform data checks before feature extraction
    df = data_checks(df, channel_columns)
    # Extract time-domain features
    time_domain_feature_df = time_domain_features(df, channel_columns)
    # # Extract frequency-domain features
    # frequency_domain_features_df = frequency_domain_features(df, channel_columns)
    # # Extract time-frequency domain features
    # time_frequency_features_df = time_frequency_features(df, channel_columns)
    return time_domain_feature_df
    # return time_domain_feature_df, frequency_domain_features_df, time_frequency_features_df




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

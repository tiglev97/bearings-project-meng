import numpy as np
import pandas as pd
import json
from scipy.fft import fft, ifft
from scipy.signal import hilbert, stft
from scipy.stats import skew, kurtosis
import pywt
import logging
import time

from pipelines.DataChecks import data_checks



def load_jsonl_to_dataframe(file_path):
    # Load JSONL file and convert it to a pandas DataFrame
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            # Flatten the nested time_series dictionary
            if 'time_series' in record:
                time_series_data = record.pop('time_series')
                record.update(time_series_data)
            data.append(record)
    df = pd.DataFrame(data)
    
    
    # Display the available columns for verification
    logging.info("Available columns in DataFrame: %s", df.columns)
    print("Available columns in DataFrame:", df.columns)

    return df


def time_domain_features(df, channel_columns=['channel x', 'channel y']):
    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():
            # Time-domain statistical features
            df[f'{channel}_mean'] = df[channel].apply(np.mean)
            df[f'{channel}_median'] = df[channel].apply(np.median)
            df[f'{channel}_std'] = df[channel].apply(np.std)
            df[f'{channel}_var'] = df[channel].apply(np.var)
            df[f'{channel}_skew'] = df[channel].apply(lambda x: skew(x))
            df[f'{channel}_kurtosis'] = df[channel].apply(lambda x: kurtosis(x))
            df[f'{channel}_rms'] = df[channel].apply(lambda x: np.sqrt(np.mean(np.square(x))))
            df[f'{channel}_ptp'] = df[channel].apply(np.ptp)  # Peak-to-Peak amplitude
            df[f'{channel}_crest_factor'] = df[channel].apply(lambda x: np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x))))
            df[f'{channel}_energy'] = df[channel].apply(lambda x: np.sum(np.square(x)))
            # Entropy can be computed using Shannon entropy
            def compute_entropy(ts):
                histogram, bin_edges = np.histogram(ts, bins=10, density=True)
                pdf = histogram / np.sum(histogram)
                pdf = pdf[pdf > 0]  # Exclude zero entries
                return -np.sum(pdf * np.log2(pdf))
            df[f'{channel}_entropy'] = df[channel].apply(compute_entropy)
        else:
            logging.error(f"Values in column '{channel}' are not in the expected list format.")
            raise ValueError(f"Values in column '{channel}' are not in the expected list format.")
    logging.info("Time-domain features extracted successfully.")
    logging.info(df)
    return df

def frequency_domain_features(df, channel_columns=['channel x', 'channel y']):
    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():
            # Apply frequency domain transformations to each time series
            df[f'{channel}_fft'] = df[channel].apply(fft)
            # Only keep the positive frequencies
            df[f'{channel}_fft_magnitude'] = df[f'{channel}_fft'].apply(lambda x: np.abs(x)[:len(x)//2])
            df[f'{channel}_fft_freq'] = df[f'{channel}_fft_magnitude'].apply(lambda x: np.fft.fftfreq(len(x)*2, d=1)[:len(x)])
            df[f'{channel}_power_spectrum'] = df[f'{channel}_fft_magnitude'].apply(lambda x: x ** 2)
            # Extract summary statistics from FFT magnitude
            df[f'{channel}_fft_mean'] = df[f'{channel}_fft_magnitude'].apply(np.mean)
            df[f'{channel}_fft_std'] = df[f'{channel}_fft_magnitude'].apply(np.std)
            df[f'{channel}_fft_max'] = df[f'{channel}_fft_magnitude'].apply(np.max)
            df[f'{channel}_fft_freq_max'] = df[f'{channel}_fft'].apply(lambda x: np.argmax(np.abs(x)))
            # Spectral centroid
            def spectral_centroid(magnitude, freq):
                return np.sum(freq * magnitude) / np.sum(magnitude)
            df[f'{channel}_spectral_centroid'] = df.apply(lambda row: spectral_centroid(row[f'{channel}_fft_magnitude'], row[f'{channel}_fft_freq']), axis=1)
            # Spectral bandwidth
            def spectral_bandwidth(magnitude, freq, centroid):
                return np.sqrt(np.sum(((freq - centroid) ** 2) * magnitude) / np.sum(magnitude))
            df[f'{channel}_spectral_bandwidth'] = df.apply(lambda row: spectral_bandwidth(row[f'{channel}_fft_magnitude'], row[f'{channel}_fft_freq'], row[f'{channel}_spectral_centroid']), axis=1)
            # Envelope Analysis
            df[f'{channel}_analytic_signal'] = df[channel].apply(hilbert)
            df[f'{channel}_amplitude_envelope'] = df[f'{channel}_analytic_signal'].apply(np.abs)
            df[f'{channel}_phase_envelope'] = df[f'{channel}_analytic_signal'].apply(np.angle)
            # Cepstrum Analysis
            df[f'{channel}_log_power_spectrum'] = df[f'{channel}_power_spectrum'].apply(lambda x: np.log(x + 1e-10))
            df[f'{channel}_cepstrum'] = df[f'{channel}_log_power_spectrum'].apply(lambda x: np.abs(ifft(x).real))
            # Extract summary statistics from cepstrum
            df[f'{channel}_cepstrum_mean'] = df[f'{channel}_cepstrum'].apply(np.mean)
            df[f'{channel}_cepstrum_std'] = df[f'{channel}_cepstrum'].apply(np.std)
            df[f'{channel}_cepstrum_max'] = df[f'{channel}_cepstrum'].apply(np.max)
        else:
            print(f"Values in column '{channel}' are not in the expected list format.")
            raise ValueError(f"Values in column '{channel}' are not in the expected list format.")
        
    logging.info("Frequency-domain features extracted successfully.")
    logging.info(df)
    return df

def time_frequency_features(df, channel_columns=['channel x', 'channel y']):
    for channel in channel_columns:
        if df[channel].apply(lambda x: isinstance(x, list)).all():
            # Time-Frequency Analysis (STFT)
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
            # Wavelet Transform (Updated)
            def compute_wavelet(ts):
                scales = np.arange(1, 128)
                wavelet = 'cmor1.0-1.5'  # Specify the bandwidth and center frequencies
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
            print(f"Values in column '{channel}' are not in the expected list format.")
            raise ValueError(f"Values in column '{channel}' are not in the expected list format.")
        
    logging.info("Time-frequency domain features extracted successfully.")
    logging.info(df)
    return df

# Combine all feature extraction functions
def extract_features(df, channel_columns=['channel x', 'channel y']):
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

import numpy as np
import logging

def data_checks(df, channel_columns=['channel x', 'channel y']):
    # Verify if the specified columns exist in the DataFrame
    missing_columns = [col for col in channel_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following specified columns are missing in the DataFrame: {missing_columns}")
    
    # Data Check 1: Check for missing values
    if df.isnull().values.any():
        logging.error("Missing values detected:")
        print("Missing values detected:")
        
        logging.error(df.isnull().sum())
        print(df.isnull().sum())
        
        # Fill missing values using forward fill
        df.fillna(method='ffill', inplace=True)
    
    # Data Check 2: Anomaly detection (using z-score)
    threshold = 3
    for column in channel_columns:
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

    # Drop the temporary columns used for anomaly detection
    df.drop(columns=[f'{column}_z_scores' for column in channel_columns], inplace=True)
    df.drop(columns=[f'{column}_anomalies' for column in channel_columns], inplace=True)

    # Return the updated DataFrame
    return df

# -*- coding: utf-8 -*-
#Import all libraries
import importlib
import subprocess
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
import ast
from sklearn.decomposition import PCA
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.cluster import DBSCAN
from tslearn.metrics import cdist_dtw


#Sample data to work with
json_file = "C:\\Users\\tigra\\OneDrive\\Documents\\cleaned_df.jsonl"

#Function to fix the data timestamp
def fix_timestamp_format(ts):
    # Check if the timestamp is in HH:MM:SS format
    parts = ts.split(":")
    # Ensure minutes and seconds have two digits each
    if len(parts) == 3:
        hour, minute, second = parts
        minute = minute.zfill(2)  #Add leading zero if missing
        second = second.zfill(2)  #Add leading zero if missing
        return f"{hour}:{minute}:{second}"
    return ts  
    


########################## Clustering function ########################

def cluster_gear_json(json_file, truncation_factor=20, regularization=0.5, mode='KMeans'):
    '''

    Parameters
    ----------
    file : str
        The JSON file which will contain the time series gear data
    truncation_factor : int
        Factor by which the data will be reduced to ensure code compiles
    regularization: float
        Factor to regularize the distance metrics
    mode: str
        'KMeans' , 'DBSCAN'

    Returns
    -------
    Distance metrics (jensen, wasserstein)

    '''
    #KMeans mode
    
    if mode == 'KMeans':
    
        #Populate pandas dataframe containing the data
        
        timestamps = []
        channel_x = []
        channel_y = []
    
        with open(json_file, 'r') as file:
            for line in file:
                try: 
                    data = json.loads(line)
                    timestamps.append(data['timestamp'])
                    channel_x.append(data['channel_x'])
                    channel_y.append(data['channel_y'])
                #print(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    
        #Make a pandas df of the data
        time_series_data = pd.DataFrame({
            'timestamp': timestamps,
            'channel_x': channel_x,
            'channel_y': channel_y
            })
        
        
        #Truncate the data
        time_series_data = time_series_data.iloc[::truncation_factor].reset_index(drop=True)
        
        
        data_x = np.array(time_series_data['channel_x'].tolist())
        data_y = np.array(time_series_data['channel_y'].tolist())
        
        # Initialize TimeSeriesKMeans with DTW metric for 4 clusters
        model_x = TimeSeriesKMeans(n_clusters=4, metric="dtw", random_state=0)
        model_y = TimeSeriesKMeans(n_clusters=4, metric="dtw", random_state=0)
        
        # Fit the model and predict clusters
        clusters_x = model_x.fit_predict(data_x)
        clusters_y = model_y.fit_predict(data_y)
        
        # Add clusters back to the DataFrame
        time_series_data['cluster_x'] = clusters_x
        time_series_data['cluster_y'] = clusters_y
        
        
        #print(time_series_data)
        
        #Adjust timestamp
        time_series_data['timestamp'] = time_series_data['timestamp'].apply(fix_timestamp_format)
        
    
        plt.figure(figsize=(12, 6))
    
        # Plot x cluster over time as dots
        plt.scatter(time_series_data['timestamp'], time_series_data['cluster_x'], label='X Cluster', color='blue', alpha=0.7)
    
        # Plot y cluster over time as dots
        plt.scatter(time_series_data['timestamp'], time_series_data['cluster_y'], label='Y Cluster', color='red', alpha=0.7)
    
        # Labels and title
        plt.xlabel("Timestamp")
        plt.ylabel("Cluster")
        plt.title("X and Y Clusters Over Time Series (DTW)")
        plt.legend()
    
        # Show plot
        plt.show()
        
        
    
    #DBSCAN Mode
    if mode == 'DBSCAN':
        
        timestamps = []
        channel_x = []
        channel_y = []

        with open(json_file, 'r') as file:
            for line in file: 
                try:
                    data = json.loads(line)
                    timestamps.append(data['timestamp'])
                    channel_x.append(data['channel_x'])
                    channel_y.append(data['channel_y'])
                    #print(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")


        time_series_data = pd.DataFrame({
            'timestamp': timestamps,
            'channel_x': channel_x,
            'channel_y': channel_y})
        
        #No truncation with DBSCAN as PCA is used
        
        data_x = np.array(time_series_data['channel_x'].tolist())
        data_y = np.array(time_series_data['channel_y'].tolist())
        
        pca = PCA(n_components=8)
        features_x = pca.fit_transform(data_x)
        features_y = pca.fit_transform(data_y)

        dbscan_x = DBSCAN(eps=0.8, min_samples=5).fit(features_x)
        dbscan_y = DBSCAN(eps=0.8, min_samples=5).fit(features_y)
        
        time_series_data['cluster_x'] = dbscan_x.labels_
        time_series_data['cluster_y'] = dbscan_y.labels_
        
        #Adjust timestamp
        time_series_data['timestamp'] = time_series_data['timestamp'].apply(fix_timestamp_format)
        
        time_series_data['timestamp'] = pd.to_datetime(time_series_data['timestamp'])

        plt.figure(figsize=(12, 6))

        # Plot x cluster over time as dots
        plt.scatter(time_series_data['timestamp'], time_series_data['cluster_x'], label='X Cluster', color='blue', alpha=0.7)

        # Plot y cluster over time as dots
        plt.scatter(time_series_data['timestamp'], time_series_data['cluster_y'], label='Y Cluster', color='red', alpha=0.7)

        # Labels and title
        plt.xlabel("Timestamp")
        plt.ylabel("Cluster")
        plt.title("X and Y Clusters Over Time Series (DBSCAN Clustering)")
        plt.legend()

        # Show plot
        plt.show()
        
        
    #Implement Jensen Metrics
    lambda_penalty = regularization

    clusters = time_series_data.groupby('cluster_x')
    
    #Represent each cluster as a probability distribution
    cluster_distributions = {}

    for cluster_label, group in clusters:
        # Flatten the `channel_x` time series data for the current cluster
        flattened_channel_x = np.concatenate(group['channel_x'].tolist())
    
        # Create a histogram for the flattened data
        hist, bin_edges = np.histogram(flattened_channel_x, bins=50, density=True)  # Normalize to a probability distribution
        cluster_distributions[cluster_label] = hist / hist.sum()  # Ensure it sums to 1


    # Calculate JSD between all pairs of clusters
    jsd_values = []
    cluster_labels = list(cluster_distributions.keys())

    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            cluster_i = cluster_distributions[cluster_labels[i]]
            cluster_j = cluster_distributions[cluster_labels[j]]
            # Calculate JSD and append to the list
            jsd = jensenshannon(cluster_i, cluster_j, base=2) ** 2  # JSD is squared to align with KL divergence units
            jsd_values.append(jsd)
            
            
    # Compute the sum of JSD values and their variance
    jsd_sum_x_cluster = sum(jsd_values)
    jsd_variance_x_cluster = np.var(jsd_values)

    # Calculate the regularized metric
    regularized_metric_x_cluster = jsd_sum_x_cluster - lambda_penalty * jsd_variance_x_cluster

    print(f"Sum of JSD values: {jsd_sum_x_cluster}")
    print(f"Variance of JSD values: {jsd_variance_x_cluster}")
    print(f"Regularized Clustering Metric: {regularized_metric_x_cluster}")
    
    # Calculate Wasserstein distance between all pairs of clusters
    wasserstein_distances = []
    cluster_labels = list(cluster_distributions.keys())

    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            cluster_i = cluster_distributions[cluster_labels[i]]
            cluster_j = cluster_distributions[cluster_labels[j]]
            # Calculate Wasserstein distance and append to the list
            w_distance = wasserstein_distance(cluster_i, cluster_j)
            wasserstein_distances.append(w_distance)
    
    
    
    # Compute the sum of Wasserstein distances and their variance
    wasserstein_sum_x_cluster = sum(wasserstein_distances)
    wasserstein_variance_x_cluster = np.var(wasserstein_distances)
    
    # Calculate the regularized metric
    regularized_metric_wasserstein_x_cluster = wasserstein_sum_x_cluster - lambda_penalty * wasserstein_variance_x_cluster

    print(f"Sum of Wasserstein distances: {wasserstein_sum_x_cluster}")
    print(f"Variance of Wasserstein distances: {wasserstein_variance_x_cluster}")
    print(f"Regularized Clustering Metric (Wasserstein) for x_cluster: {regularized_metric_wasserstein_x_cluster}")
    
    #Repeat the whole thing with y channel

    clusters = time_series_data.groupby('cluster_y')
    
    #Represent each cluster as a probability distribution
    cluster_distributions = {}
    
    for cluster_label, group in clusters:
        # Flatten the `channel_y` time series data for the current cluster
        flattened_channel_y = np.concatenate(group['channel_y'].tolist())
        
        # Create a histogram for the flattened data
        hist, bin_edges = np.histogram(flattened_channel_y, bins=50, density=True)  # Normalize to a probability distribution
        cluster_distributions[cluster_label] = hist / hist.sum()  # Ensure it sums to 1
    
    # Calculate JSD between all pairs of clusters
    jsd_values = []
    cluster_labels = list(cluster_distributions.keys())
    
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            cluster_i = cluster_distributions[cluster_labels[i]]
            cluster_j = cluster_distributions[cluster_labels[j]]
            # Calculate JSD and append to the list
            jsd = jensenshannon(cluster_i, cluster_j, base=2) ** 2  # JSD is squared to align with KL divergence units
            jsd_values.append(jsd)
    
    # Compute the sum of JSD values and their variance
    jsd_sum_y_cluster = sum(jsd_values)
    jsd_variance_y_cluster = np.var(jsd_values)
    
    # Calculate the regularized metric
    regularized_metric_y_cluster = jsd_sum_y_cluster - lambda_penalty * jsd_variance_y_cluster
    
    print(f"Sum of JSD values: {jsd_sum_y_cluster}")
    print(f"Variance of JSD values: {jsd_variance_y_cluster}")
    print(f"Regularized Clustering Metric: {regularized_metric_y_cluster}")
    print()
    
    # Calculate Wasserstein distance between all pairs of clusters
    wasserstein_distances = []
    cluster_labels = list(cluster_distributions.keys())
    
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            cluster_i = cluster_distributions[cluster_labels[i]]
            cluster_j = cluster_distributions[cluster_labels[j]]
            # Calculate Wasserstein distance and append to the list
            w_distance = wasserstein_distance(cluster_i, cluster_j)
            wasserstein_distances.append(w_distance)
    
    # Compute the sum of Wasserstein distances and their variance
    wasserstein_sum_y_cluster = sum(wasserstein_distances)
    wasserstein_variance_y_cluster = np.var(wasserstein_distances)
    
    # Calculate the regularized metric
    regularized_metric_wasserstein_y_cluster = wasserstein_sum_y_cluster - lambda_penalty * wasserstein_variance_y_cluster
    
    print(f"Sum of Wasserstein distances: {wasserstein_sum_y_cluster}")
    print(f"Variance of Wasserstein distances: {wasserstein_variance_y_cluster}")
    print(f"Regularized Clustering Metric (Wasserstein) for y_cluster: {regularized_metric_wasserstein_y_cluster}")
    
    return regularized_metric_x_cluster, regularized_metric_y_cluster, regularized_metric_wasserstein_x_cluster, regularized_metric_wasserstein_y_cluster
        


cluster_gear_json(json_file=json_file, truncation_factor=80, regularization=0.5, mode='DBSCAN')
    

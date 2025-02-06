from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import datetime

def run_clustering(cleaned_df, algorithm, params,dataset):
    # Drop the first N columns (default: 12, configurable via params)
    cleaned_df = cleaned_df.drop(cleaned_df.columns[0:12], axis=1)

    # Drop all columns with nested lists or non-scalar values
    for col in cleaned_df.columns:
        if isinstance(cleaned_df[col].iloc[0], list):
            cleaned_df.drop(col, axis=1, inplace=True)

    # Ensure all remaining columns are numeric
    cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64'])
    print(cleaned_df.columns)

    # Check if DataFrame is empty after preprocessing
    if cleaned_df.empty:
        raise ValueError("The dataset has no valid numeric columns after preprocessing.")

    # Scale the data
    scaler = StandardScaler()
    cleaned_df = scaler.fit_transform(cleaned_df)
    
    # Initialize the clustering model based on the algorithm
    if algorithm == 'DBSCAN':
        model = DBSCAN(eps=params.get('eps', 0.5), min_samples=params.get('min_samples', 5))
    elif algorithm == 'K-means':
        model = KMeans(n_clusters=params.get('n_clusters', 3), random_state=42)
    elif algorithm == 'Gaussian Mixture':
        model = GaussianMixture(n_components=params.get('n_clusters', 3), random_state=42)
    else:
        raise ValueError("Unsupported algorithm")

    # Fit the model and predict clusters
    clusters = model.fit_predict(cleaned_df)
    
    # Calculate silhouette score
    if len(set(clusters)) > 1:  # Silhouette score requires at least 2 clusters
        score = silhouette_score(cleaned_df, clusters)
    else:
        score = -1  # Assign a default invalid score for single cluster

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "algorithm": algorithm,
        "silhouette_score": score,
        "parameters": params,
        "labels": clusters.tolist()

    }

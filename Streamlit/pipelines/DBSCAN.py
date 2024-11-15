import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import ast


def PCA_DBSCAN(cleaned_df):
    # Drop the first twelve columns
    cleaned_df = cleaned_df.drop(cleaned_df.columns[0:12], axis=1)

    # Drop all columns with nested lists
    for col in cleaned_df.columns:
        if isinstance(cleaned_df[col].iloc[0], list):
            cleaned_df.drop(col, axis=1, inplace=True)

    # Ensure all remaining columns are numeric (drop any remaining non-numeric columns if necessary)
    cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64'])

    # Standardize the data
    scaler = StandardScaler()
    cleaned_df = scaler.fit_transform(cleaned_df)

    # Apply PCA to reduce dimensionality to 4 components
    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(cleaned_df)

    #print the explained variance ratio
    print(pca.explained_variance_ratio_)

    #only use pca1 and pca2

    # X_reduced = X_reduced[:, :2]
    # Fit the DBSCAN model on the PCA-reduced data
    dbscan = DBSCAN(eps=5, min_samples=6)
    clusters = dbscan.fit_predict(X_reduced)

    # seperate the clusters based on the cluster labels
    df_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    df_pca['Cluster'] = clusters

    return df_pca
# # Load the dataset
cleaned_df1 = pd.read_json("C:/uoft/Meng_project/bearings-project-meng/Streamlit/outputs/Gold/time_domain_features.jsonl", lines=True)
cleaned_df2 = pd.read_json("C:/uoft/Meng_project/bearings-project-meng/Streamlit/outputs/Gold/frequency_domain_features.jsonl", lines=True)
cleaned_df3 = pd.read_json("C:/uoft/Meng_project/bearings-project-meng/Streamlit/outputs/Gold/time_frequency_features.jsonl", lines=True)

# combine the three dataframes
cleaned_df4 = pd.concat([cleaned_df1, cleaned_df2, cleaned_df3], axis=1)


df_pca = PCA_DBSCAN(cleaned_df4)


# Plot the clusters in a scatter plot for the first two components
plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue=df_pca['Cluster'], palette='viridis', legend='full')
plt.title('DBSCAN Clustering on PCA-reduced Data (PC1 vs PC2)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



#evalueate the model
# Compute the silhouette score
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(df_pca.drop('Cluster', axis=1), df_pca['Cluster'])
print(f"Silhouette score: {silhouette}")

#-----------------DBSCAN on raw time series-----------------
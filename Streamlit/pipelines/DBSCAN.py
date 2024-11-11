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
    x_scaled = scaler.fit_transform(cleaned_df)

    # Apply PCA to reduce dimensionality to 4 components
    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(x_scaled)

    # Fit the DBSCAN model on the PCA-reduced data
    dbscan = DBSCAN(eps=1, min_samples=6)
    clusters = dbscan.fit_predict(X_reduced)

    # seperate the clusters based on the cluster labels
    df_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    df_pca['Cluster'] = clusters

    return df_pca
# # Load the dataset
cleaned_df = pd.read_json("C:/uoft/Meng_project/bearings-project-meng/Streamlit/outputs/Gold/time_domain_features.jsonl", lines=True)

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
x_scaled = scaler.fit_transform(cleaned_df)

# # Determine the cumulative explained variance for each number of PCA components
# explained_variances = []
# num_components = range(1, min(len(cleaned_df.columns), 20) + 1)  # Limit to 20 or fewer components for visualization

# for n in num_components:
#     pca = PCA(n_components=n)
#     pca.fit(x_scaled)
#     explained_variances.append(sum(pca.explained_variance_ratio_))

# # Plot the cumulative explained variance
# plt.figure(figsize=(8, 6))
# plt.plot(num_components, explained_variances, marker='o')
# plt.xlabel("Number of PCA Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.title("Elbow Method for PCA Component Selection")
# plt.grid()
# plt.show()


# Apply PCA to reduce dimensionality to 4 components
pca = PCA(n_components=6)
X_reduced = pca.fit_transform(x_scaled)



# Display explained variance ratio for PCA
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Create a DataFrame for PCA components and cluster labels for visualization
df_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
sns.pairplot(df_pca)
plt.show()

# Fit the DBSCAN model on the PCA-reduced data
dbscan = DBSCAN(eps=1, min_samples=6)
clusters = dbscan.fit_predict(df_pca)

# seperate the clusters based on the cluster labels
df_pca['Cluster'] = clusters

# Plot the clusters in a scatter plot for the first two components
plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue=clusters, palette='viridis', legend='full')
plt.title('DBSCAN Clustering on PCA-reduced Data (PC1 vs PC2)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

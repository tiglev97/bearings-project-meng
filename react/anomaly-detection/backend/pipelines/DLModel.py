import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ======== Set the Actual Input Length ========
ACTUAL_INPUT_LENGTH = 2559

# ======== Model Definitions ========

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128, input_length=ACTUAL_INPUT_LENGTH):
        super(CNNFeatureExtractor, self).__init__()
        self.input_length = input_length
        # Use ceil_mode=True so that we do not lose the extra sample.
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Compute pooled length using ceil mode twice:
        pool1_length = math.ceil(input_length / 2)
        pool2_length = math.ceil(pool1_length / 2)
        self.feature_map_length = pool2_length  
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * self.feature_map_length, feature_dim)

    def forward(self, x):
        # x: (batch, channels, length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8, num_layers=6, ff_dim=256):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, feature_dim) -> add sequence dim -> (batch, 1, feature_dim)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return x

# Clustering model: CNN + Transformer + Clustering Head.
# The clustering head outputs logits corresponding to 3 clusters.
class CNNTransformerClustering(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8, num_layers=6, num_clusters=3, input_length=ACTUAL_INPUT_LENGTH):
        super(CNNTransformerClustering, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(input_channels=1, feature_dim=feature_dim, input_length=input_length)
        self.transformer = TransformerEncoder(feature_dim=feature_dim, num_heads=num_heads, num_layers=num_layers)
        self.cluster_head = nn.Linear(feature_dim, num_clusters)  # Cluster logits.

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.transformer(features)
        logits = self.cluster_head(features)
        # Softmax gives a probability distribution over clusters.
        clusters = F.softmax(logits, dim=-1)
        return clusters, logits

# ======== Custom Dataset for Channel Y ========
# This dataset loads only the "channel_y" data (ignoring any labels).
class ChannelYDataset(Dataset):
    def __init__(self, jsonl_file):
        self.samples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Only load channel_y data.
                y = np.array(data["channel_y"])
                self.samples.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y = self.samples[idx]
        # Return channel_y as a tensor with shape (1, length)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return y_tensor

# ======== Main Execution ========
if __name__ == "__main__":
    # Path to your JSONL dataset
    dataset_file = r"C:\uoft\Meng_project\bearings-project-meng\react\anomaly-detection\backend\outputs\Silver\checked_df.jsonl"
    
    # Create the dataset (using only channel_y data)
    dataset = ChannelYDataset(dataset_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize the clustering model
    model = CNNTransformerClustering(feature_dim=128, num_heads=8, num_layers=6, num_clusters=3, input_length=ACTUAL_INPUT_LENGTH)
    model.to(device)
    
    # NOTE: A complete unsupervised deep clustering solution typically involves an iterative
    # strategy (e.g., using a clustering loss such as KL divergence with a target distribution).
    # For simplicity, here we extract features and then run K-Means as a post-processing step.
    
    model.eval()
    features_list = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            # Extract features using the CNN and Transformer components.
            features = model.feature_extractor(inputs)
            features = model.transformer(features)
            features_list.append(features.cpu().numpy())
    features_all = np.concatenate(features_list, axis=0)
    
    # Run K-Means clustering on the extracted features.
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(features_all)
    
    print("Cluster assignments:")
    print(cluster_labels)
    
    # ----- Create Chart to Display Clustering Results -----
    # Use PCA to reduce features to 2 dimensions for visualization.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_all)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title("Cluster Assignments (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster Label")
    plt.show()

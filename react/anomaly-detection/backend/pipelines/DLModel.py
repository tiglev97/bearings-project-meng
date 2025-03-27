import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print(torch.cuda.is_available())

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
        self.feature_map_length = pool2_length  # e.g., 640 for input_length 2559.
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

class Decoder(nn.Module):
    def __init__(self, feature_dim=128, output_channels=1, output_length=ACTUAL_INPUT_LENGTH):
        super(Decoder, self).__init__()
        self.output_length = output_length
        # Compute pooled length as in the encoder:
        pool1_length = math.ceil(output_length / 2)
        pool2_length = math.ceil(pool1_length / 2)
        feature_map_length = pool2_length  # e.g., 640 for output_length 2559.
        self.fc = nn.Linear(feature_dim, 128 * feature_map_length)
        # First deconvolution: output_padding=1 to help reverse the ceil_mode pooling.
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # Second deconvolution: set output_padding=0 to match the desired output length.
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=64, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=0
        )

    def forward(self, x):
        batch_size = x.size(0)
        pool1_length = math.ceil(self.output_length / 2)
        pool2_length = math.ceil(pool1_length / 2)
        feature_map_length = pool2_length
        x = self.fc(x)  # (batch, 128 * feature_map_length)
        x = x.view(batch_size, 128, feature_map_length)
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

# Single-channel autoencoder (for channel y)
class CNNTransformerAutoencoderSingleChannel(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8, num_layers=6, output_length=ACTUAL_INPUT_LENGTH):
        super(CNNTransformerAutoencoderSingleChannel, self).__init__()
        self.encoder_cnn = CNNFeatureExtractor(input_channels=1, feature_dim=feature_dim, input_length=output_length)
        self.encoder_transformer = TransformerEncoder(feature_dim, num_heads, num_layers)
        self.decoder = Decoder(feature_dim, output_channels=1, output_length=output_length)

    def forward(self, x):
        latent = self.encoder_cnn(x)
        latent = self.encoder_transformer(latent)
        recon = self.decoder(latent)
        return recon

# ======== Custom Dataset for Channel Y ========
# This dataset loads the "channel_y" data and the ground truth label from each JSON record.
class ChannelYDataset(Dataset):
    def __init__(self, jsonl_file):
        self.samples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Expect each JSON record to contain "channel_y" and optionally a "label" (0 for normal, 1 for anomaly)
                y = np.array(data["channel_y"])
                label = data.get("label", 0)
                self.samples.append((y, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, label = self.samples[idx]
        # Return channel_y as a tensor with shape (1, length)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return y_tensor, label

# ======== Training Function ========
def train_autoencoder(model, train_loader, val_loader, num_epochs=100, lr=0.0001, device="cuda"):
    model.to(device)
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Using Adam optimizer
    train_losses, val_losses = [], []
    plt.ion()  # Enable interactive plotting

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_autoencoder(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Live plotting
        plt.figure(figsize=(10, 5))
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch+2), train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch+2), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    print("Training Complete.")

# ======== Evaluation Function for Reconstruction Loss ========
def evaluate_autoencoder(model, dataloader, device="cpu"):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ======== Classification Evaluation Function ========
def evaluate_classification(model, dataloader, device="cpu", threshold=0.005):
    """
    Compute classification metrics based on a threshold on the reconstruction error.
    Reconstruction error is computed as the mean squared error per sample.
    Samples with error greater than the threshold are predicted as anomalies (label 1),
    otherwise as normal (label 0).
    """
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Compute per-sample MSE: average over all elements per sample
            batch_errors = ((outputs - inputs) ** 2).mean(dim=[1,2]).cpu().numpy()
            # Predict anomaly if error > threshold
            batch_pred = (batch_errors > threshold).astype(int)
            all_pred.extend(batch_pred.tolist())
            all_true.extend(labels)
            
    # Calculate classification metrics
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity
    }
    return metrics

# ======== Main Execution ========
if __name__ == "__main__":
    # Path to your JSONL dataset
    dataset_file = r"C:\uoft\Meng_project\bearings-project-meng\react\anomaly-detection\backend\outputs\Silver\checked_df.jsonl"
    
    # Create the dataset (using only channel_y data and ground truth labels)
    full_dataset = ChannelYDataset(dataset_file)
    dataset_size = len(full_dataset)
    
    # Split dataset into train (70%), val (15%), and test (15%)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the autoencoder model (for channel y)
    model = CNNTransformerAutoencoderSingleChannel(
        feature_dim=128, num_heads=8, num_layers=6, output_length=ACTUAL_INPUT_LENGTH
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Train the autoencoder.
    train_autoencoder(model, train_loader, val_loader, num_epochs=20, lr=0.0001, device=device)

    # Evaluate reconstruction loss on the validation and test sets.
    final_val_loss = evaluate_autoencoder(model, val_loader, device=device)
    final_test_loss = evaluate_autoencoder(model, test_loader, device=device)
    print("Final Validation Loss:", final_val_loss)
    print("Final Test Loss:", final_test_loss)

    # Evaluate classification performance on the test set.
    # NOTE: Adjust the threshold based on your data distribution.
    threshold = 0.005  
    metrics = evaluate_classification(model, test_loader, device=device, threshold=threshold)
    print("Classification Metrics on Test Set:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # Test inference: Plot original and reconstructed signal from a sample in the test set.
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            original = inputs[0].cpu().numpy().flatten()
            reconstructed = outputs[0].cpu().numpy().flatten()
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(original, label="Channel y (Original)")
            plt.title("Original Signal (Channel y)")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(reconstructed, label="Channel y (Reconstructed)")
            plt.title("Reconstructed Signal (Channel y)")
            plt.legend()
            plt.show()
            break

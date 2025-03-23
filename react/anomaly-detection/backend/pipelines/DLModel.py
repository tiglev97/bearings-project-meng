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

print(torch.cuda.is_available())
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ======== Set the Actual Input Length ========
# Your signals have length 2559.
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
        self.feature_map_length = pool2_length  # This will be 640 for input_length 2559.
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
    def __init__(self, feature_dim=128, output_channels=2, output_length=ACTUAL_INPUT_LENGTH):
        super(Decoder, self).__init__()
        self.output_length = output_length
        # Use the same calculation for pooled length as in the encoder:
        pool1_length = math.ceil(output_length / 2)
        pool2_length = math.ceil(pool1_length / 2)
        feature_map_length = pool2_length  # Should be 640 for output_length=2559.
        self.fc = nn.Linear(feature_dim, 128 * feature_map_length)
        # First deconvolution: default output_padding=1 to reverse the ceil_mode pooling.
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
        feature_map_length = pool2_length  # Expected to be 640.
        x = self.fc(x)  # (batch, 128 * feature_map_length)
        x = x.view(batch_size, 128, feature_map_length)
        x = F.relu(self.deconv1(x))  # Expected shape: (batch, 64, ?)
        x = self.deconv2(x)          # Expected shape: (batch, output_channels, output_length)
        return x

class CNNTransformerAutoencoder(nn.Module):
    def __init__(self, input_channels=2, feature_dim=128, num_heads=8, num_layers=6, output_length=ACTUAL_INPUT_LENGTH):
        super(CNNTransformerAutoencoder, self).__init__()
        self.encoder_cnn = CNNFeatureExtractor(input_channels, feature_dim, input_length=output_length)
        self.encoder_transformer = TransformerEncoder(feature_dim, num_heads, num_layers)
        self.decoder = Decoder(feature_dim, output_channels=input_channels, output_length=output_length)

    def forward(self, x):
        latent = self.encoder_cnn(x)
        latent = self.encoder_transformer(latent)
        recon = self.decoder(latent)
        return recon

# ======== Custom Dataset for Two-Channel Data ========

class TwoChannelDataset(Dataset):
    def __init__(self, jsonl_file):
        self.samples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Expect each JSON record to contain "channel_x" and "channel_y"
                x = np.array(data["channel_x"])
                y = np.array(data["channel_y"])
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # Stack the two channels into a tensor of shape (2, length)
        signal = np.stack([x, y], axis=0)
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        return signal_tensor, 0  # dummy label

# ======== Training Function for Autoencoder ========

def train_autoencoder(model, train_loader, val_loader, num_epochs=100, lr=0.0001, device="cuda"):
    model.to(device)
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        plt.plot(range(1, epoch+2), train_losses, label="Train Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch+2), val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    print("Training Complete.")

# ======== Evaluation Function ========

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

# ======== Main Execution ========

if __name__ == "__main__":
    # Path to your JSONL dataset
    dataset_file = r"C:\uoft\Meng_project\bearings-project-meng\react\anomaly-detection\backend\outputs\Silver\checked_df.jsonl"
    
    # Create the dataset and split into training and validation (80/20 split)
    full_dataset = TwoChannelDataset(dataset_file)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the autoencoder model.
    model = CNNTransformerAutoencoder(
        input_channels=2, feature_dim=128, num_heads=8, num_layers=6, output_length=ACTUAL_INPUT_LENGTH
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Train the autoencoder.
    train_autoencoder(model, train_loader, val_loader, num_epochs=20, lr=0.0001, device=device)

    # Evaluate on the validation set.
    final_val_loss = evaluate_autoencoder(model, val_loader, device=device)
    print("Final Validation Loss:", final_val_loss)

    # Example inference: Plot original and reconstructed signals for one sample.
    model.eval()
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            original = inputs[0].cpu().numpy()
            reconstructed = outputs[0].cpu().numpy()
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(original[0], label="Channel x")
            plt.plot(original[1], label="Channel y")
            plt.title("Original Signal")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(reconstructed[0], label="Channel x")
            plt.plot(reconstructed[1], label="Channel y")
            plt.title("Reconstructed Signal")
            plt.legend()
            plt.show()
            break

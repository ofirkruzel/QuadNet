import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from modelInitialize import SmallQuadNet

# Define the train_and_eval function
def train_and_eval(imu_data, gt_data):
    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values  # IMU data as features
    distance = gt_data["distance(meters)"].values  # Target variable: distance

    # Normalize the features
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    # Define the number of timesteps per sequence
    timesteps = 10  # Fixed number of timesteps
    n_features = imu_features_normalized.shape[1]

    # Synchronize lengths of features and target
    min_length = min(len(imu_features), len(distance))
    imu_features = imu_features[:min_length]
    distance = distance[:min_length]

    # Function to create sequences for time-series data
    def create_sequences(features, target, timesteps):
        X, y = [], []
        max_index = len(target) - timesteps
        for i in range(max_index):
            X.append(features[i:i + timesteps])
            y.append(target[i + timesteps])
        return np.array(X), np.array(y)

    X, y = create_sequences(imu_features_normalized, distance, timesteps)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Define the PyTorch model, loss function, and optimizer
    model = SmallQuadNet(Input=n_features, imu_window_size=timesteps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    batch_size = 32
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]

            # Adjust input shape for Conv1D
            inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Print loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        X_test = X_test.permute(0, 2, 1)  # Adjust shape for testing
        predictions = model(X_test).squeeze()
        test_loss = criterion(predictions, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.numpy(), label="True Distance", color="blue")
    plt.plot(predictions.numpy(), label="Predicted Distance", color="red", alpha=0.7)
    plt.title("True vs Predicted Distance")
    plt.xlabel("Sample")
    plt.ylabel("Distance (meters)")
    plt.legend()
    plt.grid()
    plt.show()

# Path to the parent folder containing all 27 folders
parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"

# Load and preprocess data from all folders and anlize together
merged_gt_data = []
merged_imu_data = []
for i in range(1, 2):
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)

    # Ensure the folder exists
    if os.path.isdir(folder_path):
        # Paths to GT and IMU files
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, "IMU_1.csv")

        # Load data if files exist
        if os.path.exists(gt_file) and os.path.exists(imu_file):
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)

            # Trim data to equal lengths
            min_length = min(len(gt_data), len(imu_data))
            gt_data = gt_data.iloc[:min_length]
            imu_data = imu_data.iloc[:min_length]

            # Append data
            merged_gt_data.append(gt_data)
            merged_imu_data.append(imu_data)
        else:
            print(f"Warning: Missing GT.csv or IMU_1.csv in {folder_name}")

# Merge all data into single DataFrames
final_gt_data = pd.concat(merged_gt_data, ignore_index=True)
final_imu_data = pd.concat(merged_imu_data, ignore_index=True)

# Train and evaluate the model
train_and_eval(final_imu_data, final_gt_data)

#this file does the training and evaluation of the model while marging all data files together
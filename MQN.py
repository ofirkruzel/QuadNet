import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from modelInitialize import SmallQuadNet

# Define the MQN Model




def train_and_eval(imu_data,gt_data):
       

    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values  # Use accelerometer and gyroscope data as features
    distance = gt_data["distance(meters)"].values  # Use distance as the target variable

    # Normalize 
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    # Reshape features to fit CNN input format
    # Assuming IMU data is a time-series with each timestep having 6 features
    # Define timesteps explicitly
    timesteps = 10  # Fixed number of timesteps per sequence

    def create_sequences(features, target, timesteps):
        X, y = [], []
        max_index = len(target) - timesteps
        for i in range(max_index):
            X.append(features[i:i + timesteps])
            y.append(target[i + timesteps])
        return np.array(X), np.array(y)

    # Create sequences
    X, y = create_sequences(imu_features_normalized, distance, timesteps)

    # Verify the shape of the sequences
    print(f"X shape: {X.shape}, y shape: {y.shape}")  # X: (num_samples, timesteps, features)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Verify input tensor shape
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Define model, loss, and optimizer
    model = SmallQuadNet(Input=6, imu_window_size=timesteps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    batch_size = 32
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Print loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
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

# Initialize dictionaries to hold merged GT and IMU data
merged_gt_data = []
merged_imu_data = []
# Iterate through each path_i folder (from 1 to 27)
for i in range(1, 2):
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)
    
    # Ensure the path is a directory
    if os.path.isdir(folder_path):
        # Define paths for GT and IMU_1 files
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, "IMU_1.csv")

        # Check if both files exist
        if os.path.exists(gt_file) and os.path.exists(imu_file):
            # Load the GT and IMU_1 data
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)

            # Synchronize the lengths by trimming to the shortest length
            min_length = min(len(gt_data), len(imu_data))
            gt_data = gt_data.iloc[:min_length]
            imu_data = imu_data.iloc[:min_length]

            # Add a folder identifier column (optional, for traceability)
            gt_data['Folder'] = folder_name
            imu_data['Folder'] = folder_name

            # Append GT and IMU data separately
            merged_gt_data.append(gt_data)
            merged_imu_data.append(imu_data)
        else:
            print(f"Warning: Missing GT.csv or IMU_1.csv in {folder_name}")

# Combine all merged GT and IMU data into separate DataFrames
final_gt_data = pd.concat(merged_gt_data, ignore_index=True)
final_imu_data = pd.concat(merged_imu_data, ignore_index=True)
train_and_eval(final_imu_data,final_gt_data)
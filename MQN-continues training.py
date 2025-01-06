import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from modelInitialize import SmallQuadNet
import matplotlib.pyplot as plt

# Function to save model state
def save_checkpoint(model, optimizer, iteration, checkpoint_path):
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint updated at {checkpoint_path} ")

# Function to load model state
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        if os.path.getsize(checkpoint_path) > 0:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        else:
            print("Checkpoint file is empty, starting from scratch")
            return 0
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path} ")
        return checkpoint['iteration']
    else:
        print("No checkpoint found, starting from scratch")
        return 0

#####
# Create sequences for time-series data per GT
def create_sequences(features, target, window_size,timesteps):
    X, y = [], []     
    max_index = len(target) 
    for i in range(0, max_index):
        if len(features[i:]) < window_size or i+timesteps >= max_index:
            break
        X.append(features[i:i + window_size])
        y.append(target[i+timesteps])
    return np.array(X), np.array(y)
# Define the train_and_eval function for continuous training
def train(model, optimizer, criterion,window_size,timesteps, imu_data, gt_data, iteration):
    
    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    distance = gt_data["distance(meters)"].values

    # Normalize the features
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    X, y = create_sequences(imu_features_normalized, distance, window_size,timesteps)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y ,dtype=torch.float32)

    # Training loop
    num_epochs =80
    batch_size = 32
    model.train()
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            inputs = X[i:i + batch_size]
            targets = y[i:i + batch_size]

            # Adjust input shape for Conv1D
            inputs = inputs.permute(0, 2, 1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Iteration: {iteration} | Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model


def test(model, criterion,window_size,timesteps, imu_data, gt_data, iteration):
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    distance = gt_data["distance(meters)"].values

    # Normalize the features
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    X, y = create_sequences(imu_features_normalized, distance, window_size,timesteps)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y ,dtype=torch.float32)
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        X = X.permute(0, 2, 1)  # Adjust shape for testing
        predictions = model(X).squeeze()
        test_loss = criterion(predictions, y)
        print(f"Iteration: {iteration} | Test Loss: {test_loss.item():.4f}")

        # Calculate Mean Absolute Error (MAE) as accuracy metric
        mae = torch.mean(torch.abs(predictions - y))
        print(f"Iteration: {iteration} | Test MAE: {mae.item():.4f}")

        # Append loss and MAE to lists for plotting
        losses.append(test_loss.item())
        maes.append(mae.item())
    return model

# Set up directories
losses, maes, iterations = [], [], []

parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"

# Initialize the PyTorch model, loss function, and optimizer
n_features = 6  # IMU features (fixed for all folders)
model = SmallQuadNet(Input=n_features, imu_window_size=120)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate across all folders continuously
iteration=0
for i in range(1, 23):

    iteration += 1  # Increment iteration after loading checkpoint
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)

    if os.path.isdir(folder_path):
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, "IMU_1.csv")

        if os.path.exists(gt_file) and os.path.exists(imu_file):
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)
            window_size = 120
            timesteps=int(120/(len(imu_data) / len(gt_data)))
            print(f"Training on folder {folder_path} (Iteration {iteration})")
            model = train(model, optimizer, criterion,window_size,timesteps, imu_data, gt_data, iteration)
            
        else:
            print(f"Missing data in folder {folder_path}, skipping.")
    else:
        print(f"Folder {folder_path} does not exist, skipping.")

iteration=23
for i in range(23, 28):
    
    iteration += 1  # Increment iteration after loading checkpoint
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)

    if os.path.isdir(folder_path):
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, "IMU_1.csv")

        if os.path.exists(gt_file) and os.path.exists(imu_file):
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)
            window_size = 120
            timesteps=int(120/(len(imu_data) / len(gt_data)))
            print(f"test on folder {folder_path} (Iteration {iteration})")
            model = test(model, criterion,window_size,timesteps, imu_data, gt_data, iteration)
            
        else:
            print(f"Missing data in folder {folder_path}, skipping.")
    else:
        print(f"Folder {folder_path} does not exist, skipping.")

# Plot the loss and MAE over iterations 
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.plot(maes, label='MAE')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.show()

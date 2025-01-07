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

def calculate_distance(North,East,Down):
    
    distance = []
    for i in range(1,len(North)):
        distance.append(np.sqrt((North[i]-North[i-1])**2+(East[i]-East[i-1])**2+(Down[i]-Down[i-1])**2))
    return distance

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
    North=gt_data["North"].values
    East=gt_data["East"].values
    Down=gt_data["Down"].values
    distance = calculate_distance(North,East,Down)

    # Normalize the features- consider batch normalization instead
    #scaler = StandardScaler()
    #imu_features_normalized = scaler.fit_transform(imu_features)

    X, y = create_sequences(imu_features, distance, window_size,timesteps)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y ,dtype=torch.float32)

    # Training loop
    num_epochs =70#from the article
    batch_size = 64#from the article
    model.train()
    lambda_reg = 0.01
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            inputs = X[i:i + batch_size]
            #Adjust input shape for BatchNorm1d: [batch_size, features, sequence_length]
            inputs = inputs.permute(0, 2, 1)  # Shape: [batch_size, features, sequence_length]

            # Batch normalization
            batch_norm = nn.BatchNorm1d(inputs.size(1))
            inputs_normalized = batch_norm(inputs)
            targets = y[i:i + batch_size]

            # Adjust input shape for Conv1D
            #inputs_normalized = inputs_normalized.permute(0, 2, 1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs_normalized)
            mse_loss = criterion(outputs.squeeze(), targets.squeeze())#MSE loss
            # L2 Regularization (sum of squared weights)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2) ** 2

            # Combine MSE loss with regularization
            loss = mse_loss + lambda_reg * l2_reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step(epoch_loss / len(X))

        print(f"Iteration: {iteration} | Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model


def test(model, criterion, window_size, timesteps, imu_data, gt_data, iteration):
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    North=gt_data["North"].values
    East=gt_data["East"].values
    Down=gt_data["Down"].values
    distance = calculate_distance(North,East,Down)

    # Normalize the features
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    X, y = create_sequences(imu_features_normalized, distance, window_size, timesteps)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
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

        # Calculate accuracy (percentage of predictions within a certain threshold)
        threshold = 0.1  # Define a threshold for accuracy
        accuracy = torch.mean((torch.abs(predictions - y) < threshold).float()) * 100
        print(f"Iteration: {iteration} | Test Accuracy: {accuracy.item():.2f}%")

        # Append loss, MAE, and accuracy to lists for plotting
        losses.append(test_loss.item())
        maes.append(mae.item())
        accuracies.append(accuracy.item())
    return model

# Set up directories
losses, maes, iterations,accuracies = [], [], [],[]

parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"
learning_rate = 0.001
# Initialize the PyTorch model, loss function, and optimizer
n_features = 6  # IMU features (fixed for all folders)
model = SmallQuadNet(Input=n_features, imu_window_size=120)
criterion = nn.MSELoss()
""""
# Define a regularization term (L2 regularization)
def regularized_loss(output, target, model, lambda_l2=0.01):
    mse_loss = criterion(output, target)
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)
    return mse_loss + lambda_l2 * l2_reg
"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, verbose=True)
# Train and evaluate across all folders continuously
iteration=0
for i in range(1, 27):

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

num_test=1
for i in range(27, 28):
    
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
            print("accuracy",accuracies[num_test-1])
            print("mae",maes[num_test-1])
            print("loss",losses[num_test-1])
        else:
            print(f"Missing data in folder {folder_path}, skipping.")
    else:
        print(f"Folder {folder_path} does not exist, skipping.")


from utility import *
from modelInitialize import *
from hyperparameter import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
####
def train(model, optimizer, scheduler, criterion, window_size, imu_data, gt_data, iteration, batch_size, num_epochs,lambda_reg):
    imu_features, velocity, distance, time = create_data(imu_data, gt_data)

    model = model.to(device)  # Move model to GPU
    x, v, d, t = create_sequences(imu_features, velocity, distance, time, window_size)

    print(f"Training on {len(x)} samples")
    # Training loop
    model.train()
    for epoch in range(num_epochs):

        epoch_loss = 0
        for i in range(0, len(x), batch_size):
            if i+batch_size>len(x):
                break
            inputs = x[i:i + batch_size]

            # Adjust input shape for BatchNorm1d: [batch_size, features, sequence_length]
            inputs = inputs.permute(0, 2, 1)  # Shape: [batch_size, features, sequence_length]
            inputs.to(device)
            # Batch normalization
            batch_norm = nn.BatchNorm1d(inputs.size(1)).to(device)
            inputs_normalized = batch_norm(inputs)
            # targets=v[i:i+1]
            targets = d[i:i + batch_size]  #############
            time_batch = t[i:i + batch_size]
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs_normalized)
            # mse_loss = criterion(outputs.squeeze(), targets.squeeze())#MSE loss
            mse_loss, _ = costomize_loss(time_batch, outputs.squeeze(), targets.squeeze(), criterion)
            # mse_loss=criterion(outputs.squeeze(), targets.squeeze())
            # L2 Regularization (sum of squared weights)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2) ** 2

            # Combine MSE loss with regularization
            loss = mse_loss + lambda_reg * l2_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # scheduler.step(epoch_loss / len(x))
        scheduler.step()

        print(f"Iteration: {iteration} | Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model



# Set up directories


parent_folder = r"/home/okruzelda/projects/QuadNet/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors/Horizontal"

# Initialize the PyTorch model, loss function, and optimizer
print(f"device: {device}")
model = QuadNet3(6, 120).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=optimizer_learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', scheduler_factor,scheduler_patience, verbose=True)#factor and patience from the article
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Decay by 10% every epoch
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
def lr_lambda(epoch):
    return min(1.0, (epoch + 1) / 5)  # Warmup for the first 5 epochs

#scheduler = LambdaLR(optimizer, lr_lambda)
# Train and evaluate across all folders continuously
iteration = 0
for i in range(1, 23):
    iteration += 1  # Increment iteration after loading checkpoint
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)
    if os.path.isdir(folder_path):
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, f"IMU_1.csv")
        if os.path.exists(gt_file) and os.path.exists(imu_file):
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)
            window_size = 120

            print(f"Training on folder {folder_path}|IMU 1 (Iteration {iteration})")
            model = train(model, optimizer, scheduler, criterion, window_size, imu_data, gt_data, iteration,
                              batch_size, num_epochs, loss_lambda_reg)
        else:
            print(f"Missing data in folder {folder_path}, skipping.")
    else:
        print(f"Folder {folder_path} does not exist, skipping.")

# save model parameters
torch.save(model.state_dict(), "model.pth")
# Define the file path to save the model
model_save_path = r"/home/okruzelda/projects/QuadNet/modle.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)
print(f"Model parameters saved to {model_save_path}")
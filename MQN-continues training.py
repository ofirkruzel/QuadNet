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

# Define the train_and_eval function for continuous training
def train_and_eval_continuous(model, optimizer, criterion,window_size, imu_data, gt_data, iteration, checkpoint_path):
    
    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    distance = gt_data["distance(meters)"].values

    # Normalize the features
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    #####
    # Create sequences for time-series data per GT
    def create_sequences(features, target, window_size):
        X, y = [], []
               
        max_index = len(target) 
        for i in range(1, max_index):
            if len(features[i*window_size//2:]) < window_size//2:
                break
            X.append(features[(i*window_size//2)-window_size//2:(i*window_size//2) + window_size//2])
            y.append(target[i])
        return np.array(X), np.array(y)

    X, y = create_sequences(imu_features_normalized, distance[:], window_size)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Training loop
    num_epochs = 20
    batch_size = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]

            # Adjust input shape for Conv1D
            inputs = inputs.permute(0, 2, 1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #print(f"Iteration: {iteration} | Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save model state after training
    save_checkpoint(model, optimizer, iteration, checkpoint_path)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        X_test = X_test.permute(0, 2, 1)  # Adjust shape for testing
        predictions = model(X_test).squeeze()
        test_loss = criterion(predictions, y_test)
        print(f"Iteration: {iteration} | Test Loss: {test_loss.item():.4f}")

        # Calculate Mean Absolute Error (MAE) as accuracy metric
        mae = torch.mean(torch.abs(predictions - y_test))
        print(f"Iteration: {iteration} | Test MAE: {mae.item():.4f}")

        # Append loss and MAE to lists for plotting
        losses.append(test_loss.item())
        maes.append(mae.item())
        

    return model

# Set up directories
checkpoint_path = r"C:\Users\ofirk\.vscode\ansfl\checkpoint_model\model_checkpoint.pth"
losses, maes, iterations = [], [], []
# Ensure the directory exists
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Check if the checkpoint file exists and delete it
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)  # Delete the file
    print(f"Previous checkpoint file at {checkpoint_path} deleted.")

# Create a blank file
with open(checkpoint_path, 'w') as f:
    pass  # This creates an empty file
    print(f"Blank checkpoint file created at {checkpoint_path}")
parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"

# Initialize the PyTorch model, loss function, and optimizer
n_features = 6  # IMU features (fixed for all folders)
model = SmallQuadNet(Input=n_features, imu_window_size=120)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate across all folders continuously
for i in range(1, 28):
    iteration = load_checkpoint(model, optimizer, checkpoint_path)  # Load checkpoint if available
    iteration += 1  # Increment iteration after loading checkpoint
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)

    if os.path.isdir(folder_path):
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, "IMU_1.csv")

        if os.path.exists(gt_file) and os.path.exists(imu_file):
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)
            #window_size = int((len(imu_data) / len(gt_data)) ) #how to make shur there is an equal number of sumples in each time gup? is it relavent?
            window_size = 120

            print(f"Training on folder {folder_path} (Iteration {iteration})")
            model = train_and_eval_continuous(model, optimizer, criterion,window_size, imu_data, gt_data, iteration, checkpoint_path)
            iteration += 1
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

 #להבין איך אני מריצה בתוך הרשת בצורה כזו שיהיה חלון של מדידות והGT
 #יהיה במרכז החלון ואז יהיה חפיפה בין התחומים של המדידות והרשת תלמד יותר טוב 
 
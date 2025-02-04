from utility import *
from modelInitialize import *
from hyperparameter import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd


def test(model, criterion, window_size, imu_data, gt_data, iteration, accuracy_threshold=0.1):
    imu_features, velocity, distance, time = create_data(imu_data, gt_data)
    # model = model.to(device)  # Move model to GPU

    # Normalize the features
    imu_features = imu_features.to(device)  # Ensure tensor is on GPU
    mean = imu_features.mean(dim=0, keepdim=True)  # Compute mean along columns (features)
    std = imu_features.std(dim=0, keepdim=True)  # Compute std along columns (features)
    imu_features_normalized = (imu_features - mean) / std  # Standardization formula

    x, v, d, t = create_sequences(imu_features_normalized, velocity, distance, time, window_size)
    # x,y,d,t = torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device),torch.tensor(d, dtype=torch.float32).to(device),torch.tensor(t, dtype=torch.float32).to(device)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        x = x.permute(0, 2, 1)  # Adjust shape for testing
        x.to(device)
        predictions_v = model(x)
        test_loss, prediction_d = costomize_loss(t, predictions_v.squeeze(), d.squeeze(), criterion)
        # test_loss=criterion(predictions_v.squeeze(), v.squeeze())
        # prediction_d = torch.cumsum(predictions_v, dim=0)
        print(f"Iteration: {iteration} | Test Loss: {test_loss.item():.4f}")

        # Plot prediction_d vs d
        plt.figure(figsize=(10, 6))

        # Move tensors to CPU before converting to NumPy for plotting
        plt.plot(d[1:].detach().cpu().numpy(), label="Ground Truth Distance (d)", color="blue")
        plt.plot(prediction_d.detach().cpu().numpy(), label="Predicted Distance (prediction_d)", color="orange")
        plt.xlabel("Sample")
        plt.ylabel("Distance")
        plt.title("Comparison of Predicted vs Ground Truth Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate Mean Absolute Error (MAE) as accuracy metric
        mae = torch.mean(torch.abs(prediction_d - d[1:]))
        print(f"Iteration: {iteration} | Test MAE: {mae.item():.4f}")

        rae_distance = torch.mean(torch.abs(prediction_d - d[1:]) / torch.abs(d[
                                                                              1:])) * 100  # A higher RAE indicates that, relative to the true values, your model's predictions are off by an average of the outcome
        print(f"RAE distance: {rae_distance.item():.4f}")

        # Calculate accuracy (percentage of predictions within a certain threshold)

        accuracy = torch.mean((torch.abs(prediction_d - d[
                                                        1:]) < accuracy_threshold).float()) * 100  # High accuracy suggests that the majority of predictions meet your application's requirements based on the chosen threshold.
        print(f"Iteration: {iteration} | Test Accuracy: {accuracy.item():.2f}%")

    return rae_distance, accuracy, mae, test_loss.item()


# Initialize the model
model = QuadNet3(6, 120).to(device)
criterion = nn.MSELoss()
model_save_path = r"/home/okruzelda/projects/QuadNet/modle.pth"
# Load the model parameters
model.load_state_dict(torch.load(model_save_path, weights_only=True))
print("Model parameters loaded successfully")
parent_folder = r"/home/okruzelda/projects/QuadNet/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors/Horizontal"
num_test = 1
for i in range(23, 28):

    # iteration += 1  # Increment iteration after loading checkpoint
    folder_name = f"path_{i}"
    folder_path = os.path.join(parent_folder, folder_name)

    if os.path.isdir(folder_path):
        gt_file = os.path.join(folder_path, "GT.csv")
        imu_file = os.path.join(folder_path, "IMU_1.csv")

        if os.path.exists(gt_file) and os.path.exists(imu_file):
            gt_data = pd.read_csv(gt_file)
            imu_data = pd.read_csv(imu_file)
            window_size = 120

            print(f"test on folder {folder_path} (Iteration 1)")
            accuracyies, rae, maes, losses = test(model, criterion, window_size, imu_data, gt_data, iteration=1,
                                                      accuracy_threshold=0.1)
            print(f"for test path {i}")
            print("accuracy", accuracyies)
            print("RAE", rae)
            print("mae", maes)
            print("loss", losses)
        else:
            print(f"Missing data in folder {folder_path}, skipping.")
    else:
        print(f"Folder {folder_path} does not exist, skipping.")
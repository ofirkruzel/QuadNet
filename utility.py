import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from scipy.fftpack import fft, ifft
from scipy.interpolate import UnivariateSpline
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR

from modelInitialize import QuadNet3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_distance(north, east, down):
    distance = []
    for i in range(1, len(north)):
        distance.append(
            np.sqrt((north[i] - north[i - 1]) ** 2 + (east[i] - east[i - 1]) ** 2 + (down[i] - down[i - 1]) ** 2))
    return distance


def best_smoothing_factor(time, north, east, down):
    # Split into training and validation sets
    time_train, time_val, north_train, north_val = train_test_split(time, north, test_size=0.3, shuffle=False)
    _, _, east_train, east_val = train_test_split(time, east, test_size=0.3, shuffle=False)
    _, _, down_train, down_val = train_test_split(time, down, test_size=0.3, shuffle=False)

    # Range of smoothing factors to test
    s_values = np.linspace(0, 10, 50)  # Adjust range based on data
    errors_north = []
    errors_east = []
    errors_down = []

    # Test smoothing factors for each direction
    for s in s_values:
        # Fit splines for each axis
        spline_north = UnivariateSpline(time_train, north_train, s=s)
        spline_east = UnivariateSpline(time_train, east_train, s=s)
        spline_down = UnivariateSpline(time_train, down_train, s=s)

        # Predict on validation set
        north_pred = spline_north(time_val)
        east_pred = spline_east(time_val)
        down_pred = spline_down(time_val)

        # Compute errors
        errors_north.append(mean_squared_error(north_val, north_pred))
        errors_east.append(mean_squared_error(east_val, east_pred))
        errors_down.append(mean_squared_error(down_val, down_pred))

    # Find the best smoothing factor for each direction
    best_s_north = s_values[np.argmin(errors_north)]
    best_s_east = s_values[np.argmin(errors_east)]
    best_s_down = s_values[np.argmin(errors_down)]
    """
    plt.figure(figsize=(12, 8))

    # North
    plt.subplot(2, 1, 1)
    spline_north = UnivariateSpline(time, north, s=best_s_north)
    spline_east = UnivariateSpline(time, east, s=best_s_east)
    spline_down = UnivariateSpline(time, down, s=best_s_down)
    plt.plot(time, north, 'o', label='Original North', markersize=4)
    plt.plot(time, spline_north(time), label='Smoothed North')
    plt.plot(time, east, 'o', label='Original East', markersize=4)
    plt.plot(time, spline_east(time), label='Smoothed East')
    plt.plot(time, down, 'o', label='Original Down', markersize=4)
    plt.plot(time, spline_down(time), label='Smoothed Down')
    plt.legend()
    plt.title('Trajectory (North, East, Down)')

    # Velocity
    plt.subplot(2, 1, 2)
    north_velocity = spline_north.derivative()(time)
    east_velocity = spline_east.derivative()(time)
    down_velocity = spline_down.derivative()(time)
    velocity = np.sqrt(north_velocity**2 + east_velocity**2 + down_velocity**2)
    plt.plot(time, velocity, label='Velocity')
    plt.legend()
    plt.title('Velocity')

    plt.tight_layout()
    plt.show(block=True)
    """
    ####
    return float(best_s_north), float(best_s_east), float(best_s_down)


def best_smoothing_factor_for_total_velocity(time, velocity):
    # Move tensors to CPU for NumPy calculation
    time = time.detach().cpu().numpy()
    velocity = velocity.detach().cpu().numpy()

    # Ensure sufficient data points for meaningful smoothing
    if len(time) < 10:  # Require at least 10 data points
        raise ValueError(
            f"Insufficient data points for smoothing factor selection: len(time)={len(time)}, required >= 10.")

    # Split into training and validation sets
    try:
        time_train, time_val, velocity_train, velocity_val = train_test_split(
            time, velocity, test_size=0.3, shuffle=False
        )
    except Exception as e:
        raise RuntimeError(f"Error during train-test split: {e}")

    # Ensure the train set has enough points for fitting
    if len(time_train) <= 3:
        raise ValueError(
            f"Training data too small for spline fitting: len(time_train)={len(time_train)}, required > 3."
        )

    # Define a range of smoothing factors to test
    s_values = np.linspace(0, 10, 50)  # Adjust range and resolution based on data
    errors_velocity = []

    # Test smoothing factors for velocity using UnivariateSpline
    for s in s_values:
        try:
            spline_velocity = UnivariateSpline(time_train, velocity_train, s=s)
            velocity_pred = spline_velocity(time_val)
            errors_velocity.append(mean_squared_error(velocity_val, velocity_pred))
        except Exception as e:
            # Log any errors during spline fitting or prediction
            print(f"Skipping smoothing factor s={s}: {e}")

    # Handle case where no smoothing factor produced valid results
    if not errors_velocity:
        raise RuntimeError("No valid smoothing factors tested. Check input data or smoothing range.")

    # Find the smoothing factor that minimizes the error
    best_s_velocity = s_values[np.argmin(errors_velocity)]
    return float(best_s_velocity)



def calculate_velocity_and_distance(time, north, east, down):
    best_s_north, best_s_east, best_s_down = best_smoothing_factor(time, north, east, down)
    velocity, distance = [], []
    # Fit splines to each axis with no smoothing (s=0 ensures the fit passes through all points)
    north_spline = UnivariateSpline(time, north, s=best_s_north)
    east_spline = UnivariateSpline(time, east, s=best_s_east)
    down_spline = UnivariateSpline(time, down, s=best_s_down)

    # Derive velocity functions (derivative of position)
    north_velocity = north_spline.derivative()(time)
    east_velocity = east_spline.derivative()(time)
    down_velocity = down_spline.derivative()(time)
    for i in range(1, len(north)):
        velocity.append(np.sqrt(
            (north_velocity[i] - north_velocity[i - 1]) ** 2 + (east_velocity[i] - east_velocity[i - 1]) ** 2 + (
                        down_velocity[i] - down_velocity[i - 1]) ** 2))
        distance.append(
            np.sqrt((north[i] - north[i - 1]) ** 2 + (east[i] - east[i - 1]) ** 2 + (down[i] - down[i - 1]) ** 2))
    return velocity, distance

def create_data(imu_data, gt_data):
    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    time = gt_data["time"].values
    north = gt_data["North"].values
    east = gt_data["East"].values
    down = gt_data["Down"].values

    velocity, distance = calculate_velocity_and_distance(time, north, east, down)
    return torch.tensor(imu_features, dtype=torch.float32, device=device), torch.tensor(velocity, dtype=torch.float32,
                                                                                        device=device), torch.tensor(
        distance, dtype=torch.float32, device=device), torch.tensor(time[1:], dtype=torch.float32, device=device),


# Create sequences for time-series data per GT

def create_sequences(features, velocity, distance, time, window_size):

    x, v, d, t = [], [], [], []
    max_index = len(features)
    j = 0

    # Ensure inputs like features, velocity, distance, and time are lists and not tensors
    features = features if isinstance(features, list) else features.tolist()
    velocity = velocity if isinstance(velocity, list) else velocity.tolist()
    distance = distance if isinstance(distance, list) else distance.tolist()
    time = time if isinstance(time, list) else time.tolist()

    # Calculate timesteps properly with integer conversion
    timesteps = int(120 / (len(features) / len(velocity)))
    ######
    for i in range(0, max_index):#for i in range(0, max_index, window_size):  # i is for the features, j is for the target
        if len(features[i:]) < window_size or j+timesteps >= len(velocity):
            break
        x.append(features[i:i + window_size])
        v.append(velocity[j + timesteps])
        d.append(distance[j + timesteps])
        t.append(time[j + timesteps])
        j += 1

    # Ensure tensors are returned and moved to the proper device
    return (
        torch.tensor(x, dtype=torch.float32).to(device),
        torch.tensor(v, dtype=torch.float32).to(device),
        torch.tensor(d, dtype=torch.float32).to(device),
        torch.tensor(t, dtype=torch.float32).to(device),)


def costomize_loss(time, velocity_pred, distance, criterion):
    # Use UnivariateSpline with the best smoothing factor
    best_s = best_smoothing_factor_for_total_velocity(time, velocity_pred)

    # Move tensors to the CPU if they are on GPU
    time_np = time.detach().cpu().numpy()
    velocity_pred_np = velocity_pred.detach().cpu().numpy()
    if len(time_np) != len(velocity_pred_np):
        raise ValueError(f"Length mismatch: time({len(time_np)}) vs velocity_pred({len(velocity_pred_np)})")

    # Use NumPy arrays with UnivariateSpline
    spline_velocity = UnivariateSpline(time_np, velocity_pred_np, s=best_s)

    # Compute distance predictedderivative using integration
    # Note: Ensure the range of integration matches time

    position_predicted = [spline_velocity.integral(0, t) for t in time_np]



    # Convert distance_predicted to a PyTorch tensor

    distance_predicted = [position_predicted[i] - position_predicted[i - 1] for i in range(1, len(position_predicted))]
    # Ensure distance_predicted is on the same device as the original distance tensor
    #distance_predicted = distance_predicted.to(distance.device)
    distance_predicted = torch.tensor(distance_predicted, dtype=torch.float32, device=distance.device)
    # Compute the mean squared error (MSE) loss
    mse_loss = criterion(distance_predicted, distance[1:])


    return mse_loss, distance_predicted
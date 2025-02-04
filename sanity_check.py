import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force a compatible backend (optional)
#matplotlib.use('TkAgg')

# Paths to the datasets
paths = {
    "Horizontal Trajectory": r"/home/okruzelda/projects/QuadNet/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors/Horizontal/path_1/GT.csv",
    
}

for trajectory_type, file_path in paths.items():
    print(f"Analyzing: {trajectory_type}")
    
    # Load the data
    data = pd.read_csv(file_path)
    
    # Extract relevant data
    time = data["time"]
    distance = data["distance(meters)"]
    speed = data["speed(m/s)"]
    compass_heading = data[" compass_heading(degrees)"]
    pitch = data[" pitch(degrees)"]
    roll = data[" roll(degrees)"]
    height = data["height_above_takeoff(meters)"]
    north = data["North"]
    east = data["East"]
    down = data["Down"]
    
    # Create a new figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle(f"{trajectory_type} Analysis", fontsize=16)

    # 1. 2D Horizontal Trajectory
    axs[0, 0].plot(east, north, label="2D Trajectory")
    axs[0, 0].set_xlabel("East (m)")
    axs[0, 0].set_ylabel("North (m)")
    axs[0, 0].set_title("2D Horizontal Trajectory")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # 2. Height Stability Check
    axs[0, 1].plot(time, height, label="Height Above Takeoff", color="green")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Height (m)")
    axs[0, 1].set_title("Height Stability Check")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # 3. Speed Analysis
    axs[1, 0].plot(time, speed, label="Speed", color="blue")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Speed (m/s)")
    axs[1, 0].set_title("Speed vs. Time")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # 4. Compass Heading vs. Time
    axs[1, 1].plot(time, compass_heading, label="Compass Heading", color="orange")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Heading (degrees)")
    axs[1, 1].set_title("Compass Heading vs. Time")
    axs[1, 1].legend()
    axs[1, 1].grid()

    # 5. Position in Each Axis vs. Time
    axs[2, 0].plot(time, north, label="North (N)", color="red")
    axs[2, 0].plot(time, east, label="East (E)", color="blue")
    axs[2, 0].plot(time, height, label="Altitude (Up)", color="green")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Position (m)")
    axs[2, 0].set_title("Position in Each Axis vs. Time")
    axs[2, 0].legend()
    axs[2, 0].grid()

    # 6. Velocity in Each Axis vs. Time
    axs[2, 1].plot(time, np.gradient(north, time), label="Velocity North (VN)", color="red")
    axs[2, 1].plot(time, np.gradient(east, time), label="Velocity East (VE)", color="blue")
    axs[2, 1].plot(time, np.gradient(height, time), label="Velocity Altitude (VD)", color="green")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Velocity (m/s)")
    axs[2, 1].set_title("Velocity in Each Axis vs. Time")
    axs[2, 1].legend()
    axs[2, 1].grid()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Explicitly call plt.show()
plt.show(block=True)

#can you plot a time siries discription of the exsistence of meserment in IMU_1 Vs GT in the same scale

# Show the plot

# Create a binary existence indicator for GT and IMU_1 measurements
imu_data = pd.read_csv(r"/home/okruzelda/projects/QuadNet/Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors/Horizontal/path_1/IMU_1.csv")
gt_existence = np.ones_like(time)
imu_data["time"] = pd.to_numeric(imu_data["time"], errors="coerce")
print("Rows before cleaning:", len(imu_data))
imu_data = imu_data.dropna(subset=["time"])
print("Rows after cleaning:", len(imu_data))

imu_1_time = imu_data["time"]
imu_1_existence = np.ones_like(imu_1_time)

print("GT time range:", time.min(), time.max())
print("IMU_1 time range:", imu_1_time.min(), imu_1_time.max())


# Plot existence of measurements over time
fig, ax = plt.subplots(figsize=(12, 6))
# Filter the time range to show only 0 to 0.4 seconds
time_filtered = time[(time >= 0) & (time <= 0.4)]
gt_existence_filtered = gt_existence[(time >= 0) & (time <= 0.4)]

imu_1_time_filtered = imu_1_time[(imu_1_time >= 0) & (imu_1_time <= 0.4)]
imu_1_existence_filtered = imu_1_existence[(imu_1_time >= 0) & (imu_1_time <= 0.4)]

ax.plot(time_filtered, gt_existence_filtered, label="GT Measurement Existence", color="blue", linestyle='None', marker='|')
ax.plot(imu_1_time_filtered, imu_1_existence_filtered, label="IMU_1 Measurement Existence", color="red", linestyle='None', marker='|')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Measurement Existence")
ax.set_title("Existence of IMU_1 vs GT Measurements Over Time")
ax.legend()
ax.grid()

# Show the plot
plt.show()

from scipy.interpolate import UnivariateSpline




# Fit splines to each axis with no smoothing (s=0 ensures the fit passes through all points)
north_spline = UnivariateSpline(time, north, s=0.5)
east_spline = UnivariateSpline(time, east, s=0.5)
down_spline = UnivariateSpline(time, down, s=0.5)
north_pose = north_spline(time)
# Derive velocity functions (derivative of position)
north_velocity = north_spline.derivative()(time)
east_velocity = east_spline.derivative()(time)
down_velocity = down_spline.derivative()(time)

# Velocity arrays now have the same size as the original position data
print("North Velocity:", north_velocity)
print("East Velocity:", east_velocity)
print("Down Velocity:", down_velocity)

# Plotting position and velocity for one axis (e.g., North)
plt.figure(figsize=(10, 6))
plt.plot(time, north, label='North Position', marker='o')
plt.plot(time, north_pose, label='North Position- polinom', marker='*')
plt.plot(time, north_velocity, label='North Velocity', marker='x')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Position and Velocity (North)')
plt.show()

from scipy.fftpack import fft, ifft


cutoff_freq = 0.6  
# Define a cutoff frequency to filter noise (adjust as needed)
 # Adjust based on your data characteristics
def fourier_series_approximation(time, data, cutoff_freq):
    # Ensure input is a NumPy array
    time = np.asarray(time)
    data = np.asarray(data)
 
    if len(time) < 2 or len(data) < 2:
        raise ValueError("Time and data arrays must have at least two elements.")
    
    # Fourier Transform
    freq = np.fft.fftfreq(len(time), d=(time[1] - time[0]))
    fft_coeffs = fft(data)
    
    # Filter high frequencies
    fft_coeffs_filtered = fft_coeffs * (abs(freq) < cutoff_freq)
    
    # Inverse Fourier Transform (filtered signal)
    approximated_signal = ifft(fft_coeffs_filtered).real
    
    # Compute velocity (derivative of position)
    velocity = np.gradient(approximated_signal, time)
    
    return approximated_signal, velocity

# Apply Fourier Series Approximation to each direction
north_position, north_velocity = fourier_series_approximation(time, north, cutoff_freq)
east_position, east_velocity = fourier_series_approximation(time, east, cutoff_freq)
down_position, down_velocity = fourier_series_approximation(time, down, cutoff_freq)

# Plotting Positions and Velocities
plt.figure(figsize=(12, 8))

# North
plt.subplot(3, 1, 1)
plt.plot(time, north, label='North Position (Original)', marker='o')
plt.plot(time, north_position, label='North Position (Fourier)', linestyle='--')
plt.plot(time, north_velocity, label='North Velocity', linestyle='-')
plt.legend()
plt.title('North Direction')
plt.xlabel('Time')
plt.ylabel('Value')

# East
plt.subplot(3, 1, 2)
plt.plot(time, east, label='East Position (Original)', marker='o')
plt.plot(time, east_position, label='East Position (Fourier)', linestyle='--')
plt.plot(time, east_velocity, label='East Velocity', linestyle='-')
plt.legend()
plt.title('East Direction')
plt.xlabel('Time')
plt.ylabel('Value')

# Down
plt.subplot(3, 1, 3)
plt.plot(time, down, label='Down Position (Original)', marker='o')
plt.plot(time, down_position, label='Down Position (Fourier)', linestyle='--')
plt.plot(time, down_velocity, label='Down Velocity', linestyle='-')
plt.legend()
plt.title('Down Direction')
plt.xlabel('Time')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



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

print(f"Best smoothing factor for North: {best_s_north}")
print(f"Best smoothing factor for East: {best_s_east}")
print(f"Best smoothing factor for Down: {best_s_down}")

# Fit final splines with the best smoothing factors
spline_north = UnivariateSpline(time, north, s=best_s_north) #pn
spline_east = UnivariateSpline(time, east, s=best_s_east)
spline_down = UnivariateSpline(time, down, s=best_s_down)

# Evaluate the smoothed positions
north_smoothed = spline_north(time)
east_smoothed = spline_east(time)
down_smoothed = spline_down(time)

north_velocity = spline_north.derivative()(time)#vn


# Plot original and smoothed data
plt.figure(figsize=(12, 8))

# North
plt.subplot(3, 1, 1)
plt.plot(time, north, 'o', label='Original North', markersize=4)
plt.plot(time, north_smoothed, label='Smoothed North')
plt.legend()
plt.title('North Direction')

# East
plt.subplot(3, 1, 2)
plt.plot(time, east, 'o', label='Original East', markersize=4)
plt.plot(time, east_smoothed, label='Smoothed East')
plt.legend()
plt.title('East Direction')

# Down
plt.subplot(3, 1, 3)
plt.plot(time, down, 'o', label='Original Down', markersize=4)
plt.plot(time, down_smoothed, label='Smoothed Down')
plt.legend()
plt.title('Down Direction')

plt.tight_layout()
plt.show()


def best_smoothing_factor(time,north,east,down):

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
    return float(best_s_north), float(best_s_east), float(best_s_down)
# Step 1: Fit a UnivariateSpline to the data
spline_north = UnivariateSpline(time, north, s=best_s_north)  # s=0 ensures exact fit to input data

# Step 2: Calculate the derivative of the spline (north_velocity)
spline_north_velocity = spline_north.derivative()(time)

best_ss,_,_=best_smoothing_factor(time,spline_north_velocity,east,down)

spline_north_velocity=UnivariateSpline(time, spline_north_velocity, s=best_ss)

# Step 3: Directly use the original spline for integration (spline_north contains the position function)
# Reconstructing positions at discrete data points
position_predicted = [spline_north_velocity.integral(0, t) for t in time]

# Outputs for demonstration
print("North Velocity (Derivative):", spline_north_velocity(time))
print("Position Predicted (Integral):", position_predicted)

# Plot comparison of three curves: north_velocity, position_predicted, and spline_north
plt.figure(figsize=(10, 6))
plt.plot(time, north_velocity, label='North Velocity', linestyle='-', marker='o')
plt.plot(time[:], position_predicted, label='Position Predicted north', linestyle='--', marker='x')
plt.plot(time, spline_north(time), label='Spline position North', linestyle=':', marker='*')
plt.legend()
plt.title('Comparison of North Velocity, Position Predicted, and Spline North')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
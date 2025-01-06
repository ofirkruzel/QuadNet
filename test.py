import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Force a compatible backend (optional)
import matplotlib
matplotlib.use('TkAgg')

# Paths to the datasets
paths = {
    "Horizontal Trajectory": r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal\path_1\GT.csv",
    
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
imu_data = pd.read_csv(r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal\path_1\IMU_1.csv")
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
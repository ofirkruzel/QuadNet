import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Force a compatible backend (optional)
import matplotlib
matplotlib.use('TkAgg')

# Paths to the datasets
paths = {
    "Horizontal Trajectory": r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal\path_1\GT.csv",
    "Straight Line": r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\StraightLine\path_1\GT.csv",
    "Vertical Trajectory": r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Vertical\path_1\GT.csv",
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

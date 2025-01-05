import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Define the MQN Model

def build_quadnet_model(input_shape):
    model = tf.keras.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same", input_shape=input_shape))#layer 1+2: 120*64*1
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="same"))#layer 3+4:
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding="same"))

    # Flatten the output
    model.add(tf.keras.layers.Flatten())

    # Fully connected layers
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1))  # Regression output

    # Compile the model with TensorFlow's Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae'])

    return model


def train_and_eval(imu_data,gt_data):
    
    

    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values  # Use accelerometer and gyroscope data as features
    distance = gt_data["distance(meters)"].values  # Use distance as the target variable

    # Normalize 
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    # Reshape features to fit CNN input format
    # Assuming IMU data is a time-series with each timestep having 6 features
    timesteps = 10  # Number of timesteps per sample (hyperparameter)
    n_features = imu_features_normalized.shape[1]
    # Synchronize lengths of features and target
    min_length = min(len(imu_features), len(distance))
    imu_features = imu_features[:min_length]
    distance = distance[:min_length]

    # Updated create_sequences function
    def create_sequences(features, target, timesteps):
        X, y = [], []
        max_index = len(target) - timesteps  # Prevent out-of-bounds indices
        for i in range(max_index):
            X.append(features[i : i + timesteps])
            y.append(target[i + timesteps])
        return np.array(X), np.array(y)

    X, y = create_sequences(imu_features_normalized, distance, timesteps)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_samples_x,timesteps_x,feature_x=X_train.shape
    model1=build_quadnet_model(input_shape=(timesteps_x,feature_x))

    # Compile the model
    model1.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, mae = model1.evaluate(X_test, y_test, verbose=2)
    print(f"for the basic 3 layer modle- Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # Predict distances
    y_pred = model1.predict(X_test)

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True Distance", color="blue")
    plt.plot(y_pred, label="Predicted Distance", color="red", alpha=0.7)
    plt.title("basic 3 layer model-True vs Predicted Distance")
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
for i in range(1, 28):
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
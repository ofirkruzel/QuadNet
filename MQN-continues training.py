import os

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
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR

from modelInitialize import QuadNet3


def calculate_distance(north,east,down):
    
    distance = []
    for i in range(1,len(north)):
        distance.append(np.sqrt((north[i]-north[i-1])**2+(east[i]-east[i-1])**2+(down[i]-down[i-1])**2))
    return distance

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

def best_smoothing_factor_for_total_velocity(time,velocity):
    velocity = velocity.detach().numpy()
    # Split into training and validation sets
    time_train, time_val, velocity_train, velocity_val = train_test_split(time, velocity, test_size=0.3, shuffle=False)
    
    # Range of smoothing factors to test
    s_values = np.linspace(0, 10, 50)  # Adjust range based on data
    errors_velocity = []

    # Test smoothing factors for each direction
    for s in s_values:
        # Fit splines for each axis
        spline_velocity = UnivariateSpline(time_train, velocity_train, s=s)

        # Predict on validation set
        velocity_pred = spline_velocity(time_val)

        # Compute errors
        errors_velocity.append(mean_squared_error(velocity_val, velocity_pred))

    # Find the best smoothing factor for each direction
    best_s_velocity = s_values[np.argmin(errors_velocity)]

    return float(best_s_velocity)

def calculate_velocity_and_distance(time,north,east,down):
    best_s_north, best_s_east, best_s_down=best_smoothing_factor(time,north,east,down)
    velocity,distance=[],[]
    # Fit splines to each axis with no smoothing (s=0 ensures the fit passes through all points)
    north_spline = UnivariateSpline(time, north, s=best_s_north)
    east_spline = UnivariateSpline(time, east, s=best_s_east)
    down_spline = UnivariateSpline(time, down, s=best_s_down)

    # Derive velocity functions (derivative of position)
    north_velocity = north_spline.derivative()(time)
    east_velocity = east_spline.derivative()(time)
    down_velocity = down_spline.derivative()(time)
    for i in range(1,len(north)):
        velocity.append(np.sqrt((north_velocity[i]-north_velocity[i-1])**2+(east_velocity[i]-east_velocity[i-1])**2+(down_velocity[i]-down_velocity[i-1])**2))
        distance.append(np.sqrt((north[i]-north[i-1])**2+(east[i]-east[i-1])**2+(down[i]-down[i-1])**2))
    return velocity,distance

def create_data(imu_data, gt_data):
    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    time=gt_data["time"].values
    north=gt_data["North"].values
    east=gt_data["East"].values
    down=gt_data["Down"].values
  
    velocity, distance = calculate_velocity_and_distance(time,north,east,down)
    return imu_features,velocity, distance,time[1:]


# Create sequences for time-series data per GT
def create_sequences(features, velocity,distance,time, window_size):
    x, v,d,t = [], [],[],[]
    max_index = len(features) 
    j=0
    timesteps=int(120/(len(features) / len(velocity)))
    for i in range(0, max_index,window_size): # i is for the features, j is for the target
        if len(features[i:]) < window_size or j >= len(velocity):
            break
        x.append(features[i:i + window_size])
        v.append(velocity[j+timesteps])#i+timesteps]
        d.append(distance[j+timesteps])#i+timesteps]
        t.append(time[j+timesteps])
        j+=1
    return np.array(x), np.array(v),np.array(d),np.array(t)


def costomize_loss(time,velocity_pred, distance, criterion):
    # Define the train_and_eval function for continuous training
    # Use UnivariateSpline with the best smoothing factor
    best_s = best_smoothing_factor_for_total_velocity(time,velocity_pred)
    time_np = time.detach().numpy()
    velocity_pred_np = velocity_pred.detach().numpy()
    if len(time_np) != len(velocity_pred_np):
        raise ValueError(f"Length mismatch: time({len(time_np)}) vs velocity_pred({len(velocity_pred_np)})")

    # Use NumPy arrays with UnivariateSpline
    spline_velocity = UnivariateSpline(time_np, velocity_pred_np, s=best_s)
    # Integrate to get position
    #position = spline_velocity.antiderivative()(time)
    # Compare to GT position - distance
    distance_predicted = [spline_velocity.integral(time[i - 1], time[i]) for i in range(1, len(time))]
    # Calculate MSE loss for position
    #mse_loss = criterion(torch.tensor(position, dtype=torch.float32), torch.tensor(distance, dtype=torch.float32))
    #mse_loss = criterion(position.clone().detach().float(), distance.clone().detach().float())
    #sample from position len(distance)+1 data set' and the calculate the delta between evert 2 samples
    
    distance_predicted = torch.tensor(distance_predicted, dtype=torch.float32)

    import matplotlib.pyplot as plt
    
    # Plotting distance_predicted vs time
    plt.figure(figsize=(10, 6))
    plt.plot(time.detach().numpy()[1:], distance_predicted.numpy(), label="Predicted Distance", color="red")
    plt.xlabel("Time")
    plt.ylabel("Distance Predicted")
    plt.title("Distance Predicted vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    mse_loss = criterion(distance_predicted.float(), distance[1:].float())
    
    return mse_loss,distance_predicted


def train(model, optimizer,scheduler, criterion,window_size, imu_data, gt_data, iteration,batch_size,num_epochs,lambda_reg):
    imu_features, velocity,distance,time= create_data(imu_data, gt_data)
    
    #plot the velocity in time for the GT data
    #plt.plot(GT_reg_data)
    #plt.title("velocity in time")
    #plt.show()

    x, y,d,t = create_sequences(imu_features, velocity,distance,time, window_size)
    x, y ,d,t= torch.tensor(x, dtype=torch.float32), torch.tensor(y ,dtype=torch.float32),torch.tensor(d ,dtype=torch.float32),torch.tensor(t ,dtype=torch.float32)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        for i in range(0, len(x), batch_size):
            inputs = x[i:i + batch_size]
            #Adjust input shape for BatchNorm1d: [batch_size, features, sequence_length]
            inputs = inputs.permute(0, 2, 1)  # Shape: [batch_size, features, sequence_length]

            # Batch normalization
            batch_norm = nn.BatchNorm1d(inputs.size(1))
            inputs_normalized = batch_norm(inputs)
            targets = d[i:i + batch_size]
            time_batch=t[i:i + batch_size]
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs_normalized)
            #mse_loss = criterion(outputs.squeeze(), targets.squeeze())#MSE loss
            mse_loss,_=costomize_loss(time_batch,outputs.squeeze(), targets.squeeze(), criterion)
            
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

        #scheduler.step(epoch_loss / len(X))
        scheduler.step()


        print(f"Iteration: {iteration} | Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model


def test(model, criterion, window_size, imu_data, gt_data, iteration,accuracy_threshold = 0.1):
    imu_features, velocity,distance,time= create_data(imu_data, gt_data)

    # Normalize the features
    scaler = StandardScaler()
    imu_features_normalized = scaler.fit_transform(imu_features)

    x,y,d,t = create_sequences(imu_features_normalized, velocity,distance,time, window_size)
    x,y,d,t = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),torch.tensor(d, dtype=torch.float32),torch.tensor(t, dtype=torch.float32)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        
        x = x.permute(0, 2, 1)  # Adjust shape for testing
        predictions_v = model(x).squeeze()
        test_loss,prediction_d = costomize_loss(t,predictions_v.squeeze(), d.squeeze(), criterion)
        print(f"Iteration: {iteration} | Test Loss: {test_loss.item():.4f}")
        
        # Plot prediction_d vs d
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(d[1:].detach().numpy(), label="Ground Truth Distance (d)", color='blue')
        plt.plot(prediction_d.detach().numpy(), label="Predicted Distance (prediction_d)", color='orange')
        plt.xlabel("Sample")
        plt.ylabel("Distance")
        plt.title("Comparison of Predicted vs Ground Truth Distance")
        plt.legend()
        plt.grid(True)
        plt.show()
    
        # Calculate Mean Absolute Error (MAE) as accuracy metric
        mae = torch.mean(torch.abs(prediction_d - d[1:]))
        print(f"Iteration: {iteration} | Test MAE: {mae.item():.4f}")

        rae_distance = torch.mean(torch.abs(prediction_d - d[1:])/torch.abs(d[1:]))*100#A higher RAE indicates that, relative to the true values, your model's predictions are off by an average of the outcome
        print(f"RAE distance: {rae_distance.item():.4f}")

        # Calculate accuracy (percentage of predictions within a certain threshold)
        
        accuracy = torch.mean((torch.abs(prediction_d - d[1:]) < accuracy_threshold).float()) * 100#High accuracy suggests that the majority of predictions meet your application's requirements based on the chosen threshold.
        print(f"Iteration: {iteration} | Test Accuracy: {accuracy.item():.2f}%")

        
    return rae_distance, accuracy, mae, test_loss.item()




def real_training():
    # Set up directories
    #hyperparameters
    optimizer_learning_rate = 0.00001
    loss_lambda_reg = 0.01#in the training loop
    num_epochs =70#from the article
    batch_size = 64#from the article
    scheduler_factor=0.7
    scheduler_patience=4
    accuracy_threshold = 0.1

    parent_folder = r"C:\Users\ofirk\PycharmProjects\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"

    # Initialize the PyTorch model, loss function, and optimizer
    model = QuadNet3(6, 120)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=optimizer_learning_rate)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', scheduler_factor,scheduler_patience, verbose=True)#factor and patience from the article
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Decay by 10% every epoch
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    def lr_lambda(epoch):
        return min(1.0, (epoch + 1) / 5)  # Warmup for the first 5 epochs

    scheduler = LambdaLR(optimizer, lr_lambda)
    # Train and evaluate across all folders continuously
    iteration=0
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
                model = train(model, optimizer,scheduler, criterion,window_size, imu_data, gt_data, iteration,batch_size,num_epochs,loss_lambda_reg)        
            else:
                print(f"Missing data in folder {folder_path}, skipping.")
        else:
            print(f"Folder {folder_path} does not exist, skipping.")
    
    #save model parameters
    torch.save(model.state_dict(), "model.pth")
    # Define the file path to save the model
    model_save_path = r"C:\Users\ofirk\PycharmProjects\ansfl\modle_parameters.pth"

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")

def real_testing(): 
    # Initialize the model
    model = QuadNet3(6, 120)
    criterion = nn.MSELoss()
    model_save_path = r"C:\Users\ofirk\PycharmProjects\ansfl\modle_parameters.pth"
    # Load the model parameters
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    print("Model parameters loaded successfully")
    parent_folder = r"C:\Users\ofirk\PycharmProjects\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"
    num_test=1
    for i in range(23, 28):
        
        #iteration += 1  # Increment iteration after loading checkpoint
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
                accuracyies,rae,maes,losses  = test(model, criterion,window_size, imu_data, gt_data, iteration=1,accuracy_threshold=0.1)
                print(f"for test path {i}")
                print("accuracy",accuracyies)
                print("RAE",rae)
                print("mae",maes)
                print("loss",losses)
            else:
                print(f"Missing data in folder {folder_path}, skipping.")
        else:
            print(f"Folder {folder_path} does not exist, skipping.")

#optuna_stady()
#real_run()
#real_training()
real_testing()
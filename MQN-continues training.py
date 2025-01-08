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
import optuna

def calculate_distance(North,East,Down):
    
    distance = []
    for i in range(1,len(North)):
        distance.append(np.sqrt((North[i]-North[i-1])**2+(East[i]-East[i-1])**2+(Down[i]-Down[i-1])**2))
    return distance

def create_data(imu_data, gt_data):
    # Extract relevant features (IMU data) and target (distance)
    imu_features = imu_data[["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]].values
    North=gt_data["North"].values
    East=gt_data["East"].values
    Down=gt_data["Down"].values
    distance = calculate_distance(North,East,Down)
    return imu_features, distance

# Create sequences for time-series data per GT
def create_sequences(features, target, window_size,timesteps):
    X, y = [], []     
    max_index = len(features) 
    j=0
    for i in range(0, max_index,window_size): # i is for the features, j is for the target
        if len(features[i:]) < window_size or j >= len(target):
            break
        X.append(features[i:i + window_size])
        y.append(target[j+timesteps])#i+timesteps]
        j+=1
    return np.array(X), np.array(y)

# Define the train_and_eval function for continuous training
def train(model, optimizer,scheduler, criterion,window_size,timesteps, imu_data, gt_data, iteration,batch_size,num_epochs,lambda_reg):
    imu_features, distance= create_data(imu_data, gt_data)

    X, y = create_sequences(imu_features, distance, window_size,timesteps)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y ,dtype=torch.float32)

    # Training loop
    model.train()
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


def test(model, criterion, window_size, timesteps, imu_data, gt_data, iteration,accuracy_threshold = 0.1):
    imu_features, distance= create_data(imu_data, gt_data)

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

        RAE_distance = torch.mean(torch.abs(predictions - y)/torch.abs(y))*100#A higher RAE indicates that, relative to the true values, your model's predictions are off by an average of the outcome
        print(f"RAE distance: {RAE_distance.item():.4f}")

        # Calculate accuracy (percentage of predictions within a certain threshold)
        
        accuracy = torch.mean((torch.abs(predictions - y) < accuracy_threshold).float()) * 100#High accuracy suggests that the majority of predictions meet your application's requirements based on the chosen threshold.
        print(f"Iteration: {iteration} | Test Accuracy: {accuracy.item():.2f}%")

        
    return RAE_distance, accuracy, mae, test_loss.item()


def objective(trial):
    # Set up directories
    losses, maes, iterations,accuracies = [], [], [],[]
    #hyperparameters
    optimizer_learning_rate =  trial.suggest_float("optimizer_learning_rate", 0.001, 0.2)
    loss_lambda_reg = trial.suggest_float("lambda_reg", 0.001, 0.2)#in the training loop
    num_epochs =trial.suggest_int("num_epochs", 50, 90)#in the training loop#from the article 70
    batch_size =trial.suggest_int("batch_size", 30, 80)#from the article 64
    #scheduler_factor=trial.suggest_float("scheduler_factor", 0.1, 0.9)#from the article 0.7
    scheduler_factor=0.7
    #scheduler_patience=trial.suggest_int("scheduler_patience", 1,10)#from the article 4
    scheduler_patience=4
    accuracy_threshold=0.1

    parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"

    # Initialize the PyTorch model, loss function, and optimizer
    model = SmallQuadNet(6, 120)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=optimizer_learning_rate)
    scheduler = scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
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
                timesteps=int(120/(len(imu_data) / len(gt_data)))
                #print(f"Training on folder {folder_path}|IMU 1 (Iteration {iteration})")
                model = train(model, optimizer, scheduler,criterion,window_size,timesteps, imu_data, gt_data, iteration,batch_size,num_epochs,loss_lambda_reg)        
            else:
                print(f"Missing data in folder {folder_path}, skipping.")
        else:
            print(f"Folder {folder_path} does not exist, skipping.")

    num_test=1
    for i in range(23, 28):
        
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
                #print(f"test on folder {folder_path} (Iteration {iteration})")
                accuracy,mae,loss = test(model, criterion,window_size,timesteps, imu_data, gt_data, iteration,accuracy_threshold)
                #print("accuracy",accuracies[num_test-1])
                #print("mae",maes[num_test-1])
                #print("loss",losses[num_test-1])
            else:
                print(f"Missing data in folder {folder_path}, skipping.")
        else:
            print(f"Folder {folder_path} does not exist, skipping.")
    print ("accuracy",accuracy)    
    return loss


def optuna_stady():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    trial = study.best_trial

    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def real_training():
    # Set up directories
    #hyperparameters
    optimizer_learning_rate = 0.001
    loss_lambda_reg = 0.01#in the training loop
    num_epochs =70#from the article
    batch_size = 64#from the article
    scheduler_factor=0.7
    scheduler_patience=4
    accuracy_threshold = 0.1

    parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"

    # Initialize the PyTorch model, loss function, and optimizer
    model = SmallQuadNet(6, 120)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=optimizer_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', scheduler_factor,scheduler_patience, verbose=True)#factor and patience from the article

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
                timesteps=int(120/(len(imu_data) / len(gt_data)))
                print(f"Training on folder {folder_path}|IMU 1 (Iteration {iteration})")
                model = train(model, optimizer,scheduler, criterion,window_size,timesteps, imu_data, gt_data, iteration,batch_size,num_epochs,loss_lambda_reg)        
            else:
                print(f"Missing data in folder {folder_path}, skipping.")
        else:
            print(f"Folder {folder_path} does not exist, skipping.")
    
    #save model parameters
    torch.save(model.state_dict(), "model.pth")
    # Define the file path to save the model
    model_save_path = r"C:\Users\ofirk\.vscode\ansfl\modle_parameters.pth"

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")

def real_testing(): 
    # Initialize the model
    model = SmallQuadNet(6, 120)
    criterion = nn.MSELoss()
    model_save_path = r"C:\Users\ofirk\.vscode\ansfl\modle_parameters.pth"
    # Load the model parameters
    model.load_state_dict(torch.load(model_save_path))
    print("Model parameters loaded successfully")
    parent_folder = r"C:\Users\ofirk\.vscode\ansfl\Quadrotor-Dead-Reckoning-with-Multiple-Inertial-Sensors\Horizontal"
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
                timesteps=int(120/(len(imu_data) / len(gt_data)))
                print(f"test on folder {folder_path} (Iteration 1)")
                accuracyies,RAE,maes,losses  = test(model, criterion,window_size,timesteps, imu_data, gt_data, iteration=1,accuracy_threshold=0.1)
                print(f"for test path {i}")
                print("accuracy",accuracyies)
                print("RAE",RAE)
                print("mae",maes)
                print("loss",losses)
            else:
                print(f"Missing data in folder {folder_path}, skipping.")
        else:
            print(f"Folder {folder_path} does not exist, skipping.")

#optuna_stady()
#real_run()
real_training()
real_testing()
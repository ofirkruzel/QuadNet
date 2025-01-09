import torch.nn as nn
import torch.nn.functional as F

# define Artur neural network model
class QuadNet3(nn.Module):
    # notice that we inherit from nn.Module
    def __init__(self, Input=6, imu_window_size=120):
        super(QuadNet3, self).__init__()
        # self.imu_window_size = imu_window_size
        # self.input = Input
        # self.conv0 = nn.Conv1d(6, 32, 3, 1, padding=1)
        # self.bn0 = nn.BatchNorm1d(32)
        # self.conv1 = nn.Conv1d(self.input, 64, 3, 1, padding=1)
        # self.conv_1_2 = nn.Conv1d(64, 64, 3, 1, padding=1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.conv2 = nn.Conv1d(64, 128, 3, 1, padding=1)
        # self.conv_2_3 = nn.Conv1d(128, 128, 3, 1, padding=1)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(128, 256, 3, 1, padding=1)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.conv_3_4 = nn.Conv1d(256, 256, 3, 1, padding=1)
        # self.conv4 = nn.Conv1d(256, 512, 3, 1, padding=1)
        # self.conv_4_5 = nn.Conv1d(512, 512, 3, 1, padding=1)
        # self.pool = nn.MaxPool1d(2, 2)
        # self.fc1 = nn.Linear(self.imu_window_size * 512, 4096)
        # self.fc_1_2 = nn.Linear(4096, 4096)
        # self.fc2 = nn.Linear(4096, 512)
        # self.fc3 = nn.Linear(512, 1)
        # self.dropout2 = nn.Dropout(0.8)

        self.imu_window_size = imu_window_size
        self.Input = Input
        self.conv0 = nn.Conv1d(6, 32, 3, 1, padding=1)
        self.bn0 = nn.BatchNorm1d(32)
        self.conv1 = nn.Conv1d(self.Input, 64, 3, 1, padding=1)
        self.conv_1_2 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, 1, padding=1)
        self.conv_2_3 = nn.Conv1d(128, 128, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv_3_4 = nn.Conv1d(256, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, 3, 1, padding=1)
        # self.conv_4_5 = nn.Conv1d(512, 512, 3, 1, padding=1)
        # self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(self.imu_window_size * 512, 4096)
        # self.fc_1_2 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout2 = nn.Dropout(0.8)

    def forward(self, x):
        # 1st Layer
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv_1_2(x)
        x = F.leaky_relu(x)
        # 2nd Layer
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv_2_3(x)
        x = F.leaky_relu(x)
        # 3rd Layer
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv_3_4(x)
        x = F.leaky_relu(x)
        # 4th Layer
        x = self.conv4(x)
        x = F.leaky_relu(x)
        # FC layers
        x = x.view(-1, self.imu_window_size * 512)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)

        return self.fc3(x)


class SmallQuadNet(nn.Module):
    # notice that we inherit from nn.Module
    def __init__(self, Input=6, imu_window_size=120):
        super(SmallQuadNet, self).__init__()

        self.imu_window_size = imu_window_size
        self.Input = Input
        self.conv0 = nn.Conv1d(6, 32, 3, 1, padding=1)
        self.bn0 = nn.BatchNorm1d(32)
        self.conv1 = nn.Conv1d(self.Input, 64, 3, 1, padding=1)
        self.conv_1_2 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, 1, padding=1)
        self.conv_2_3 = nn.Conv1d(128, 128, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv_3_4 = nn.Conv1d(256, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, 3, 1, padding=1)
        self.fc1 = nn.Linear(self.imu_window_size * 512, 512)
        self.fc2 = nn.Linear(512, 1)
        #self.dropout2 = nn.Dropout(0.8)
        # Dropout Layers
        self.dropout1 = nn.Dropout(0.5)  # Add dropout after some convolutional layers
        self.dropout2 = nn.Dropout(0.8)  # Existing dropout
        self.dropout3 = nn.Dropout(0.5)  # Add dropout after the FC layers

    def forward(self, x):
        #print("Input shape to SmallQuadNet forward:", x.shape) 
        # 1st Layer
        x = self.conv1(x)
        x = F.leaky_relu(x)
        #x=self.dropout1(x)#########
        x = self.conv_1_2(x)
        x = F.leaky_relu(x)
        # 2nd Layer
        x = self.conv2(x)
        x = F.leaky_relu(x)
        #x=self.dropout1(x)#########
        x = self.conv_2_3(x)
        x = F.leaky_relu(x)
        # 3rd Layer
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv_3_4(x)
        x = F.leaky_relu(x)
        #x=self.dropout1(x)##########
        # 4th Layer
        x = self.conv4(x)
        x = F.leaky_relu(x)
        # FC layers
        x = x.view(-1, self.imu_window_size * 512)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x))
        #x = self.dropout3(x) 

        return x
    
    
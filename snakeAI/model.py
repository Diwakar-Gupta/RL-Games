import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(out_features = 24, in_features = input_size)
        self.linear2 = nn.Linear(out_features = 48, in_features = 24)
        self.linear3 = nn.Linear(out_features = 24, in_features = 48)
        self.linear4 = nn.Linear(out_features = output_size, in_features = 24)

        self.gamma = 0.9
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.criterion = nn.MSELoss()
#        atexit.register(self.save)

    def save(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
    
    def predict(self, x):
        return self.forward(x)
    
    def train_step(self, target, pred):
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
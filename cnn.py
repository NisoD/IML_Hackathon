import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

class CNNRegressor(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, dropout_rate=0.2):
        super(CNNRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class CNNTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
           

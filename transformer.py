import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)
        return x

class TransformerTrainer:
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
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(test_loader)

def transformer_network(X_train, X_test, y_train, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    input_dim = X_train.shape[1]
    d_model = 128
    nhead = 8
    num_layers = 4

    # Prepare data
    train_loader, test_loader = prepare_data(X_train, X_test, y_train, y_test, batch_size)

    # Initialize model, loss, and optimizer
    model = TransformerRegressor(input_dim, d_model, nhead, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize trainer
    trainer = TransformerTrainer(model, criterion, optimizer, device)

    # Training loop
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Final evaluation
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(batch_y.numpy())

    y_pred = np.clip(np.array(y_pred).flatten(), 0, 50).astype(int)
    y_true = np.array(y_true).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    print(f"Transformer Network - MSE: {mse}")
    
    return y_true, y_pred

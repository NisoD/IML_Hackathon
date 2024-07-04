import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
     
    return test_loss / len(test_loader), predictions, actuals, mse, mae, r2

def convert_to_pytorch(X_train, X_test, y_train, y_test):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_dataloader(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor):
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def train_on_all_models(X_train, X_test, y_train, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_to_pytorch(X_train, X_test, y_train, y_test)
   
    # Create DataLoader
    train_loader, test_loader = create_dataloader(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

    # Initialize model
    model = SimpleNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.L1Loss()  # Correctly instantiate the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(num_epochs)):
        train_model(model, train_loader, criterion, optimizer, device)
        test_loss, predictions, actuals, mse, mae, r2 = evaluate_model(model, test_loader, criterion, device)
        print(f'Final Loss: {test_loss:.4f}')
        print(f'Final MSE: {mse:.4f}')
        print(f'Final MAE: {mae:.4f}')
        print(f'Final R2 Score: {r2:.4f}')
        train_losses.append(criterion(model(X_train_tensor.to(device)), y_train_tensor.to(device)).item())
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}')

    # Final evaluation
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f'Final MSE: {mse:.4f}')
    print(f'Final R2 Score: {r2:.4f}')
    print(f'Accuracy:{r2*100}%')

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()


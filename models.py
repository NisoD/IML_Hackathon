import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
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
def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2, "Linear Regression"

def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2, "Random Forest"

def gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2, "Gradient Boosting"


# def adaboost(X_train, X_test, y_train, y_test):
#     model = AdaBoostRegressor(n_estimators=50, random_state=42)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     return mse, mae, r2, "AdaBoost"

def compare_models(X_train, X_test, y_train, y_test):
    models = [
        linear_regression,
        random_forest,
        gradient_boosting,
    ]
    
    results = []
    for model_func in models:
        mse, _, r2, name = model_func(X_train, X_test, y_train, y_test)
        results.append({
            'name': name,
            'MSE': mse,
            'R2': r2
        })
        print(f"{name} - Loss (MSE): {mse:.4f}, Accuracy (R²): {r2:.4f}")
    
    return results

def train_on_all_models(X_train, X_test, y_train, y_test):
    
    print("Evaluating traditional ML models:")
    ml_results = compare_models(X_train, X_test, y_train, y_test)
   
   # Then, train and evaluate the neural network
    print("\nTraining Neural Network:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_to_pytorch(X_train, X_test, y_train, y_test)
    train_loader, test_loader = create_dataloader(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

    model = SimpleNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(num_epochs)):
        train_model(model, train_loader, criterion, optimizer, device)
        test_loss, predictions, actuals, mse, mae, r2 = evaluate_model(model, test_loader, criterion, device)
        train_losses.append(criterion(model(X_train_tensor.to(device)), y_train_tensor.to(device)).item())
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}')

    print(f'Neural Network - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Accuracy: {r2*100:.2f}%')

    # Plot loss curves for neural network
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Training and Test Loss')
    plt.legend()
    plt.show()

    all_results = ml_results + [{
        'name': 'Neural Network',
        'MSE': mse,
        'R2': r2
    }]

    # Prepare data for plotting
    model_names = [result['name'] for result in all_results]
    accuracies = [result['R2'] for result in all_results]
    losses = [result['MSE'] for result in all_results]

    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies)
    plt.ylabel('Accuracy (R²)')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot loss comparison
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, losses)
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


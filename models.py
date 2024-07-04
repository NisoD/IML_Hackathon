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
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from neural_network import neural_network
from typing import Tuple

def pred_is_0_to_50(y_pred: np.ndarray) -> np.ndarray:
    # Clip the values to be between 0 and 50 and convert to integers
    y_pred = np.clip(y_pred, 0, 50).astype(int)
    return y_pred


def linear_regression(X_train, X_test, y_train, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    #round as predictions are integers 0-50 as bus seats in real life
    y_pred = pred_is_0_to_50(np.round(model.predict(X_test)))
    MSE = mean_squared_error(y_test, y_pred)
    print(model.__class__.__name__, MSE)
    return y_test, y_pred 



def random_forest(X_train, X_test, y_train, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = pred_is_0_to_50(np.round(model.predict(X_test)))
    MSE = mean_squared_error(y_test, y_pred)
    print(model.__class__.__name__, MSE)
    return y_test, y_pred 


def gradient_boosting(X_train, X_test, y_train, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = pred_is_0_to_50(np.round(model.predict(X_test)))
    MSE = mean_squared_error(y_test, y_pred)
    print(model.__class__.__name__, MSE)
    return y_test, y_pred 


def adaboost(X_train, X_test, y_train, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = AdaBoostRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = pred_is_0_to_50(np.round(model.predict(X_test)))
    MSE = mean_squared_error(y_test, y_pred)
    print(model.__class__.__name__, MSE)
    return y_test, y_pred 

def create_two_csv_files(trip_id_unique_station :pd, y_test :np.ndarray, y_pred: np.ndarray, model_name: str):
    # create two csv in directory - passengares_up
    # if not available mkdir
    if not os.path.exists('passengers_up'):
        os.makedirs('passengers_up')

    df = pd.DataFrame({'trip_id_unique_station': trip_id_unique_station, 'passengers_up': y_pred})
    df.to_csv(f'passengers_up/passengers_up_predictions_{model_name}.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'passengers_up/y_test_{model_name}.csv', index=False)

def train_on_all_models(X_train, X_test, y_train, y_test):
    models = [
        # linear_regression,
        # random_forest,
        # gradient_boosting,
        # adaboost,
        neural_network
    ]
    
    trip_id_unique_station_test = X_test["trip_id_unique_station"]
    X_test = X_test.drop(columns=["trip_id_unique_station"])
    X_train = X_train.drop(columns=["trip_id_unique_station"])
   
    for model in models:
        y_test, y_pred = model(X_train, X_test, y_train, y_test)
        model_name = model.__name__
        create_two_csv_files(trip_id_unique_station_test, y_test, y_pred, model_name)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model: {model_name}")
        print(f"MSE: {mse}")
        print("=====================================")
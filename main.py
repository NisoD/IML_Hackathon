import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset, random_split
import click
import pprint
from sklearn.linear_model import LinearRegression


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-8')


def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

def save_data(train, test):
    train.to_csv('./data/train.csv', index=False)
    test.to_csv('./data/test.csv', index=False)

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def main(file_path):
    data = load_data(file_path)
    print(data.shape)
    data.drop(['part', 'cluster', 'station_name','door_closing_time','trip_id_unique_station','trip_id_unique','alternative','arrival_time'], axis=1, inplace=True)
    train, test = split_data(data)
    #label is passengares up
    logging.info("Data saved to train.csv and test.csv")

    X_test = test.drop(['passengers_up'], axis=1)
    X_train = train.drop(['passengers_up'], axis=1)
    y_test = test['passengers_up']
    y_train = train['passengers_up']


    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Linear Regression MSE: {lr_mse}')
    plt.plot()
    plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label='Predicted vs Actual')
    plt.scatter(y_test, y_test, c='red', alpha=0.5, label='Actual values')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

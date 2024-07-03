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
import random


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

    data['alternative'] = data['alternative'].astype('category').cat.codes
    data['station_name'] = data['station_name'].astype('category').cat.codes
    # handle time data:
    data.drop(['part', 'cluster','trip_id_unique_station','trip_id_unique','arrival_time'], axis=1, inplace=True)
    data.drop(['mekadem_nipuach_luz'], axis=1, inplace=True ) # didn't change accuracy
    data.drop(['longitude','latitude'], axis=1, inplace=True ) # 0.01 down accuracy
    # data.drop(['arrival_is_estimated'], axis=1, inplace=True ) # very importand!
    """calculate diffrence of time between the arrival time and the time of the previous station"""

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
   # Select a random subset of 250 points
    indices = random.sample(range(len(y_test)), 250)
    y_test_subset = y_test.values[indices]
    y_pred_subset = y_pred[indices]

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Linear Regression MSE: {lr_mse}')
    print("accuracy",lr_model.score(X_test, y_test)*100,"%")


if __name__ == '__main__':
    main()
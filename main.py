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
import click
import pprint

def load_data(filename, test_size=0.2):
    logging.info(f"Loading data from {filename}")
    data = pd.read_csv(filename)
    print(data.head())
    logging.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Split the data
    train, test = train_test_split(data, test_size=test_size)
    logging.info(f"Data split into train ({train.shape[0]} rows) and test ({test.shape[0]} rows)")
    
    # Save the split data to CSV files
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    logging.info("Train and test data saved to 'train.csv' and 'test.csv' respectively")

@click.command()
@click.option('--training-set', type=click.Path(exists=True), required=True, help="Path to the training set")
@click.option('--test-set', type=click.Path(), required=False, help="Path to the test set")
@click.option('--out', type=click.Path(), required=False, help="Output directory")
def main(training_set, test_set=None, out=None):
    logging.basicConfig(level=logging.INFO)
    
    # Load and process the data
    load_data(training_set)

if __name__ == '__main__':
    main()

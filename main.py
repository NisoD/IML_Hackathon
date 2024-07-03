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
    train, test = split_data(data)
    save_data(train, test)
    logging.info("Data saved to train.csv and test.csv")


if __name__ == '__main__':
    main()

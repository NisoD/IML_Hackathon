import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import click
from sklearn.linear_model import LinearRegression
from typing import Tuple
from models import train_on_all_models
from sklearn.preprocessing import StandardScaler
import click
from models import train_on_all_models
"""#####################################
-- usage :python main.py data/train.csv
"""#####################################

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path:str)-> pd.DataFrame:
    # return pd.read_csv(file_path, encoding='ISO-8859-8')
    return pd.read_csv(file_path, encoding='utf8')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import click
from sklearn.linear_model import LinearRegression
from typing import Tuple
from models import train_on_all_models
from sklearn.preprocessing import StandardScaler
import click
from models import train_on_all_models
"""#####################################
-- usage :python main.py data/train.csv
"""#####################################

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path:str)-> pd.DataFrame:
    # return pd.read_csv(file_path, encoding='ISO-8859-8')
    return pd.read_csv(file_path, encoding='utf8')

def split_data_problem_1(data: pd.DataFrame):
    X = data.drop('passengers_up', axis=1)
    y = data['passengers_up']
    
    # Extract trip_id_unique_station before splitting
    trip_id_unique_station = X["trip_id_unique_station"].copy()
    X = X.drop("trip_id_unique_station", axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Add trip_id_unique_station back to the scaled DataFrames
    X_train_scaled['trip_id_unique_station'] = trip_id_unique_station[X_train.index]
    X_test_scaled['trip_id_unique_station'] = trip_id_unique_station[X_test.index]
    
    return X_train_scaled, X_test_scaled, y_train, y_test
def accuracy(y_test, y_pred, epsilon=0.1):
    return np.mean(np.abs(y_test - y_pred) < epsilon) * 100
#unused only for init
def save_data(train, test:Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train.to_csv('./data/train.csv', index=False)
    test.to_csv('./data/test.csv', index=False)

def create_time_diff(data:pd.DataFrame)-> pd.DataFrame:
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S')
    data['time_difference'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds() / 60
    data['time_difference'] = data['time_difference'].clip(lower=0)
    return data


def drop_unused_data(data:pd.DataFrame)-> pd.DataFrame:
    data.drop(['trip_id', 'trip_id_unique', 
                       'arrival_time', 'door_closing_time', 'mekadem_nipuach_luz', 
                       'passengers_continue_menupach'], axis=1, inplace=True)
    return data
    
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    data = create_time_diff(data)
    data = drop_unused_data(data)
    #  boolean column to numeric
    data['arrival_is_estimated'] = data['arrival_is_estimated'].map({True: 1, False: 0}).fillna(0).astype(int)

    # Convert categorical variables to numeric
    categorical_cols = [ 'direction', 'alternative', 'cluster', 'station_name','part']
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes
    # Fill NaN values with 0 FOR NOW CHANGE IT TO MEAN OR SOMETHING ELSE
    ## DONT FORGET TO CHANGE IT ##############################################
    data = data.fillna(0)
    return data
    
def passengers_data_split(train,test:Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train.drop(['passengers_up'], axis=1)
    X_test = test.drop(['passengers_up'], axis=1)
    y_train = train['passengers_up']
    y_test = test['passengers_up']
    return X_train, X_test, y_train, y_test

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def main(file_path):
    data = load_data(file_path)
    data = preprocess(data)
    X_train_scaled, X_test_scaled, y_train, y_test = split_data_problem_1(data)
    train_on_all_models(X_train_scaled, X_test_scaled, y_train, y_test)

if __name__ == '__main__':
    main()

def accuracy(y_test, y_pred, epsilon=0.1):
    return np.mean(np.abs(y_test - y_pred) < epsilon) * 100
#unused only for init
def save_data(train, test:Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train.to_csv('./data/train.csv', index=False)
    test.to_csv('./data/test.csv', index=False)

def create_time_diff(data:pd.DataFrame)-> pd.DataFrame:
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S')
    data['time_difference'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds() / 60
    data['time_difference'] = data['time_difference'].clip(lower=0)
    return data


def drop_unused_data(data:pd.DataFrame)-> pd.DataFrame:
    data.drop(['trip_id', 'trip_id_unique', 
                       'arrival_time', 'door_closing_time', 'mekadem_nipuach_luz', 
                       'passengers_continue_menupach'], axis=1, inplace=True)
    return data
    
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    data = create_time_diff(data)
    data = drop_unused_data(data)
    #  boolean column to numeric
    data['arrival_is_estimated'] = data['arrival_is_estimated'].map({True: 1, False: 0}).fillna(0).astype(int)

    # Convert categorical variables to numeric
    categorical_cols = [ 'direction', 'alternative', 'cluster', 'station_name','part']
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes
    # Fill NaN values with 0 FOR NOW CHANGE IT TO MEAN OR SOMETHING ELSE
    ## DONT FORGET TO CHANGE IT ##############################################
    data = data.fillna(0)
    return data
    
def passengers_data_split(train,test:Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train.drop(['passengers_up'], axis=1)
    X_test = test.drop(['passengers_up'], axis=1)
    y_train = train['passengers_up']
    y_test = test['passengers_up']
    return X_train, X_test, y_train, y_test

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def main(file_path):
    data = load_data(file_path)
    data = preprocess(data)
    X_train_scaled, X_test_scaled, y_train, y_test = split_data_problem_1(data)
    train_on_all_models(X_train_scaled, X_test_scaled, y_train, y_test)

if __name__ == '__main__':
    main()
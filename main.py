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
from sklearn.preprocessing import LabelEncoder
"""#####################################
-- usage :python main.py data/train.csv
"""#####################################
RANDOM_STATE = 42
WORDS_WEIGHT = ['תחנה מרכזית', 'רכבת', 'קניון', 'מסוף', 'ת. מרכזית', 'בי"ח', 'אוניברסיטה']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path:str)-> pd.DataFrame:
    return pd.read_csv(file_path, encoding='ISO-8859-8')
    # return pd.read_csv(file_path, encoding='utf8')

def split_data_problem_1(data: pd.DataFrame):
    X = data.drop('passengers_up', axis=1)
    y = data['passengers_up']
    
    trip_id_unique_station = X["trip_id_unique_station"].copy()
    X = X.drop("trip_id_unique_station", axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    X_train_scaled['trip_id_unique_station'] = trip_id_unique_station[X_train.index]
    X_test_scaled['trip_id_unique_station'] = trip_id_unique_station[X_test.index]
    
    return X_train_scaled, X_test_scaled, y_train, y_test

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
    
def calculate_approx_line_length(data):
    first_station = data.iloc[0]
    last_station = data.iloc[-1]
    return np.sqrt((last_station['latitude'] - first_station['latitude'])**2 + 
                   (last_station['longitude'] - first_station['longitude'])**2)

def preprocess_duration_problem(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # arrival_time to datetime
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S')
    # calc trip duration
    trip_durations = data.groupby('trip_id_unique').agg({
        'arrival_time': ['min', 'max']
    })
    trip_durations.columns = ['start_time', 'end_time']
    trip_durations['trip_duration_minutes'] = (trip_durations['end_time'] - trip_durations['start_time']).dt.total_seconds() / 60
    # Feature engineering 
    features = pd.DataFrame()
    features['station_cnt'] = data.groupby("trip_id_unique")["trip_id_unique_station"].nunique()
    features['total_passenger'] = data.groupby("trip_id_unique")["passengers_up"].sum()
    features['mean_passenger'] = data.groupby("trip_id_unique")["passengers_up"].mean()
    features['mean_passenger_continue'] = data.groupby("trip_id_unique")["passengers_continue"].mean()
    features['start_hour'] = data.groupby("trip_id_unique")["arrival_time"].min().dt.hour
    
    # Add cluster, direction, and mekadem_nipuach_luz
    features = features.merge(data[["trip_id_unique", "cluster", "direction", "mekadem_nipuach_luz"]].drop_duplicates(), on="trip_id_unique")
    
    # Encode 
    label_encoder = LabelEncoder()
    features['cluster'] = label_encoder.fit_transform(features['cluster'])
    
    # Add station name 
    station_concat = data.groupby("trip_id_unique")["station_name"].agg(lambda x: ', '.join(x)).reset_index()
    for i, word in enumerate(WORDS_WEIGHT, start=1):
        features[f'station_{word}'] = station_concat['station_name'].str.contains(word, regex=False).astype(int)
    
    #  aaprox  line length
    features['line_length_approx'] = data.groupby('trip_id_unique').apply(calculate_approx_line_length)
    
    #  trip durations 
    features = features.merge(trip_durations['trip_duration_minutes'], on='trip_id_unique')
    
    # Separate features and labels
    X = features.drop(['trip_duration_minutes', 'trip_id_unique'], axis=1)
    y = features['trip_duration_minutes']
    
    return X, y

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

def trip_duration_split(train,test):
    # trip duration =  max(station_index) arrival time - arg min(station index) arrival time
    trip_duration_train = train.groupby('trip_id_unique_station')['arrival_time'].max() - train.groupby('trip_id_unique_station')['arrival_time'].min()
    trip_duration_test = test.groupby('trip_id_unique_station')['arrival_time'].max() - test.groupby('trip_id_unique_station')['arrival_time'].min()

def passengers_data_split(train,test:Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train.drop(['passengers_up'], axis=1)
    X_test = test.drop(['passengers_up'], axis=1)
    y_train = train['passengers_up']
    y_test = test['passengers_up']
    return X_train, X_test, y_train, y_test
@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def main(file_path):
    # data = load_data(file_path)
    # data = preprocess(data)
    # X_train_scaled, X_test_scaled, y_train, y_test = split_data_problem_1(data)
    # train_on_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
    data = load_data(file_path)
    
    #Problem 1 
    data = preprocess(data)
    X_train, X_test, y_train, y_test = split_data_problem_1(data)
    train_on_all_models(X_train, X_test, y_train, y_test,problem_num=1)
    
    #Problem 2
    X, y = preprocess_duration_problem(data)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=RANDOM_STATE)
    train_on_all_models(X_train, X_test, y_train, y_test,problem_num=2)
if __name__ == '__main__':
    main()

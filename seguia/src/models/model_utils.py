import pandas as pd
from src.data.utils import (
    get_general_path, join_paths
)

RANDOM_SEED = 69
MIN_TRAINING_DATE = '2017-01-01'
MAX_TRAINING_DATE = '2023-06-05'
TARGET = 'num_drought_index_future'
PROCESSED_DATA = 'data/processed/'
MODEL_PATH = 'models/'


def read_xy_set(set_name):
    general_path = get_general_path()
    x_filename = f'x_{set_name}.parquet'
    y_filename = f'y_{set_name}.parquet'
    x_path = join_paths(general_path, PROCESSED_DATA, x_filename)
    y_path = join_paths(general_path, PROCESSED_DATA, y_filename)
    x = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    y = y[TARGET]
    return x, y


def get_model(model_name='SeGuia.pkl'):
    general_path = get_general_path()
    model_filepath = join_paths(general_path, MODEL_PATH, model_name)
    model = pd.read_pickle(model_filepath)
    return model


def predict(model, x_set):
    prediction = model.predict_proba(x_set)[:, 1]
    return prediction

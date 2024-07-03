import pandas as pd
import json
import numpy as np


def get_data(path):
    """Get data, induce index, transform dates"""
    data = pd.read_csv(path, index_col=0)
    for col in data.columns:
        if 'DATE' in col:
            data[col] = pd.to_datetime(data[col])
    return data


def process_str_data_to_json_requirements(str_data):
    return str_data.replace("'", '"').replace('None', '-9999')


def read_as_dict(string_dict, specific_key=None):
    string_dict_process = process_str_data_to_json_requirements(string_dict)
    dict_as_dict = json.loads(string_dict_process)
    if specific_key:
        return dict_as_dict.get(specific_key)
    return dict_as_dict


def get_dataframe_from_dict(dictionary):
    return pd.DataFrame(dictionary).replace(-9999, np.nan)

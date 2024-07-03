import datetime
import os
import yaml
import pickle


def get_general_path():
    '''Function to get the general path'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    general_path = os.path.join(file_path, '..', '..')
    return general_path


def join_paths(*p1):
    """
    Helper function to join paths
    """
    return os.path.join(*p1)


def check_if_filepath_exists(filepath):
    """Check if the corresponding path exists."""
    exists = os.path.exists(filepath)
    return exists


def save_as_pickle(what, where):
    """
    Helper function to save a file `what` into the path `where` as a pickle.
    """
    print(f'Saving file into: "{where}".')
    with open(where, 'wb') as file:
        pickle.dump(what, file)
    print('Done saving file.')


def make_desired_folder(data_file_path):
    general_path = get_general_path()
    file_path = join_paths(general_path, data_file_path)
    exists = check_if_filepath_exists(file_path)
    if not exists:
        os.makedirs(file_path)
    return None


def get_reference_url():
    '''Get the urls from the config file'''
    general_path = get_general_path()
    file_path = os.path.join(general_path, 'config', 'config.conf')
    with open(file_path, 'r') as file:
        information = yaml.load(file, Loader=yaml.SafeLoader)
    return information


def get_data_path(name):
    '''Obtain the relative path for the data folder'''
    general_path = get_general_path()
    file_path = join_paths(general_path, 'data', name)
    return file_path


def path_for_raw_file(name):
    raw_data_path = get_data_path('raw')
    raw_file_path = os.path.join(raw_data_path, name)
    return raw_file_path


def save_dataframe(filepath, dataframe, file_format='parquet'):
    if file_format == 'parquet':
        dataframe.to_parquet(filepath)
    elif file_format == 'csv':
        dataframe.to_csv(filepath)
    elif file_format == 'pickle':
        dataframe.to_pickle(filepath)
    print(f'Data was saved into `{filepath}`.')


def create_index__mun_id(data, set_index=True):
    df = data.copy()
    entity_component = df.CVE_ENT.apply(zeroes_to_cve, zeroes=2)
    municipal_component = df.CVE_MUN.apply(zeroes_to_cve, zeroes=3)
    df['mun_id'] = entity_component + '_' + municipal_component
    if set_index:
        df.set_index('mun_id', inplace=True)
    return df


def zeroes_to_cve(cve_digit, zeroes):
    str_cve_digit = str(cve_digit)
    str_cve_digit_with_front_zeroes = str_cve_digit.zfill(zeroes)
    return str_cve_digit_with_front_zeroes


def get_time_from_string(string_date):
    date = datetime.datetime.strptime(string_date, '%Y%m%d')
    return date

import networkx as nx
import os
import pandas as pd

from src.data.utils import (
    get_general_path,
    join_paths,
    save_as_pickle,
    create_index__mun_id
)
from src.features.neighbour_utils import get_intersections_as_list
from src.features.meteorological_utils import (
    obtain_df_from_meteorological_data
)
from src.features.drought_utils import (
    transform_numerical_drought_index,

)

RAW_DATA_PATH = 'data/raw'
INTERIM_DATA_PATH = 'data/interim'
MUNICIPAL_DATA_FILE = 'municipal_data.pkl'
GRAPH_FILE = 'municipal_neighbour_graph.pkl'
METEO_RAW_DATA = 'data/raw/meteorological_info/'
METEOROLOGICAL_FILE = "meteorological_information"
TARGET_DATA_FILE = 'target_datasets.pkl'
DROUGTH_DATA_FILE = 'drought_data.parquet'
PROCESSED_DROUGHT_DATA_FILE = 'processed_drought_data.parquet'
TIME_IDENTIFIER = '00:00'
DATE_NAME = 'date'
NEW_DATE = 'standard_date'
THRESHOLD_DATE = '2016-01-01'
TARGET_NAME = 'num_drought_index'
COMPARISION_DAY = 18
PARTITION_COL = ['mun_id']


def joining_meteorological_data():
    print('Executing joining_meteorological_data function...')
    general_path = get_general_path()
    municipal_data_path = join_paths(
        general_path, INTERIM_DATA_PATH, MUNICIPAL_DATA_FILE
    )
    meteorological_data_path = join_paths(general_path, METEO_RAW_DATA)
    final_meteorological_data_path = join_paths(
        general_path, INTERIM_DATA_PATH, METEOROLOGICAL_FILE
    )
    print('Getting municipal data...')
    municipal_data = pd.read_pickle(municipal_data_path)
    meteorological_files = [
        file.split('.')[0]
        for file in os.listdir(meteorological_data_path)
    ]
    mun_ids = sorted(
        list(
            set(meteorological_files).intersection(set(municipal_data.index))
        )
    )

    meteo_dfs = [
        obtain_df_from_meteorological_data(mun_id) for mun_id in mun_ids
    ]
    joined_meteo_dfs = pd.concat(meteo_dfs)

    pt1 = joined_meteo_dfs[joined_meteo_dfs.mun_id <= '08_']

    pt2 = joined_meteo_dfs[(
            (joined_meteo_dfs.mun_id > '08_') &
            (joined_meteo_dfs.mun_id <= '16_')
    )]
    pt3 = joined_meteo_dfs[(
            (joined_meteo_dfs.mun_id > '16_') &
            (joined_meteo_dfs.mun_id <= '22_')
    )]
    pt4 = joined_meteo_dfs[joined_meteo_dfs.mun_id > '22_']

    print(
        f'Saving meteorological data as '
        f'parquets at: {final_meteorological_data_path}'
    )
    pt1.to_parquet(
        final_meteorological_data_path,
        partition_cols=PARTITION_COL
    )
    print('Done with first part, continue with second...')
    pt2.to_parquet(
        final_meteorological_data_path,
        partition_cols=PARTITION_COL
    )
    print('Done with second part, continue with third...')
    pt3.to_parquet(
        final_meteorological_data_path,
        partition_cols=PARTITION_COL
    )
    print('Done with third part, continue with fourth...')
    pt4.to_parquet(
        final_meteorological_data_path,
        partition_cols=PARTITION_COL
    )
    print(f'Done saving data. Used {PARTITION_COL}  as partition column. ')


def identify_neighbours():
    general_path = get_general_path()
    municipal_data_path = join_paths(
        general_path, INTERIM_DATA_PATH, MUNICIPAL_DATA_FILE
    )
    municipal_data = pd.read_pickle(municipal_data_path)
    municipal_data['neighbours'] = municipal_data.geometry.apply(
        get_intersections_as_list, df=municipal_data
    )
    reduced_municipal_data = municipal_data.neighbours.explode().reset_index()
    reduced_municipal_data = reduced_municipal_data[
        reduced_municipal_data.mun_id != reduced_municipal_data.neighbours
    ]
    neighbours_graph = nx.from_pandas_edgelist(
        reduced_municipal_data, 'mun_id', 'neighbours'
    )

    graph_file = join_paths(general_path, INTERIM_DATA_PATH, GRAPH_FILE)
    save_as_pickle(what=neighbours_graph, where=graph_file)


def process_drought_data():
    general_path = get_general_path()
    drougth_data_path = join_paths(
        general_path, RAW_DATA_PATH, DROUGTH_DATA_FILE
    )
    drought_data = pd.read_parquet(drougth_data_path)
    drought_data = create_index__mun_id(drought_data, set_index=False)
    not_date_columns = [
        col for col in drought_data.columns
        if not (TIME_IDENTIFIER in str(col))
    ]
    data = pd.melt(
        drought_data,
        id_vars=not_date_columns,
        var_name=DATE_NAME,
        value_name=TARGET_NAME
    )
    data[DATE_NAME] = pd.to_datetime(data[DATE_NAME])

    timedelta = pd.to_timedelta(data[DATE_NAME].dt.day, 'd')

    data.loc[data[DATE_NAME].dt.day > COMPARISION_DAY, NEW_DATE] = (
        data[DATE_NAME] - timedelta + pd.to_timedelta(28, 'd')
    )
    data.loc[data[DATE_NAME].dt.day <= COMPARISION_DAY, NEW_DATE] = (
        data[DATE_NAME] - timedelta + pd.to_timedelta(15, 'd')
    )
    data[TARGET_NAME] = data[TARGET_NAME].apply(
        transform_numerical_drought_index
    )
    date_as_string = data[NEW_DATE].dt.strftime('%Y%m%d')
    data['mun_id__date'] = data['mun_id'] + '__' + date_as_string
    data.set_index('mun_id__date', inplace=True)
    procesed_drougth_data_path = join_paths(
        general_path, INTERIM_DATA_PATH, PROCESSED_DROUGHT_DATA_FILE
    )
    print(f'Saving data into: {procesed_drougth_data_path}')
    data.to_parquet(procesed_drougth_data_path)
    print('Done saving the data.')

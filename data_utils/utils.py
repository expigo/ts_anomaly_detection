import os
from pathlib import PurePath, Path
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.neighbors import KernelDensity
import seaborn as sns

import data_utils.datasets as datasets

def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_dir_path():
    return get_project_root().joinpath('data')


def get_trained_models_dir_path():
    return get_project_root().joinpath('trained_models')


def get_all_filenames(data_dir_path: Path) -> dict:
    filenames = next(os.walk(data_dir_path), (None, None, []))[2]
    return filenames


def get_all_dirnames(data_dir_path: Path) -> dict:
    dir_names = next(os.walk(data_dir_path), (None, None, []))[1]
    return dir_names


def get_hexagon_ts_metadata(filenames):
    return {f.split(sep='_')[0]: (f,
                                  int(f.split('_')[4]),
                                  int(f.split('_')[5]),
                                  int(f.split('_')[6].split('.')[0]))
            for f in filenames}


def get_hexagon_ts_by_id(id, data_dir_path=get_data_dir_path()) -> (np.array, int):
    path_to_hexagon = data_dir_path.joinpath('hexagon_anomaly')
    filenames = get_all_filenames(path_to_hexagon)
    metadata = get_hexagon_ts_metadata(filenames)
    file_path, split_index, anomaly_start_idx, anomaly_stop_idx = metadata[f'{id:03}']
    original_id = file_path.split(sep='_')[0]
    return np.genfromtxt(PurePath(path_to_hexagon).joinpath(file_path), delimiter='\n').reshape(-1, 1), \
           split_index, anomaly_start_idx, anomaly_stop_idx, original_id


def plot_hexagon_location_by_id(id, vlines=[], show=False):
    data, split_index, anomaly_start_idx, anomaly_stop_idx, original_id = get_hexagon_ts_by_id(id)

    fig, ax = plt.subplots(figsize=(12, 6))
    # plt.figure(2)
    ax.plot(data, color='royalblue')
    ax.vlines([split_index, *vlines], *ax.get_ylim(), linestyles='dashed', colors='k', label='train/test split')
    ax.axvspan(anomaly_start_idx, anomaly_stop_idx, alpha=0.5, color='salmon', label='anomaly occurrence area')

    ax.set_title(f'location {original_id}\n split @ {split_index}')
    plt.legend()
    if show:
        plt.show()


import matplotlib
matplotlib.use("TkAgg")

# 90 is interesting but big
plot_hexagon_location_by_id(22, show=True)


def get_chosen_hexagon_ids():
    return [
        1,
        2,
        3,
        4,
        11,
    ]

def get_all_hexagon_ids(data_dir_path=get_data_dir_path().joinpath('hexagon_anomaly')):
    filenames = get_all_filenames(data_dir_path)
    metadata = get_hexagon_ts_metadata(filenames)
    return metadata.keys()


def get_hexagon_descriptions(ids=[]):
    descriptions_dfs = []
    if not ids:
        path_to_hexagon = get_data_dir_path().joinpath('hexagon_anomaly')
        filenames = get_all_filenames(path_to_hexagon)
        for filename in filenames:
            descriptions_dfs.append(pd.DataFrame(
                np.genfromtxt(PurePath(path_to_hexagon).joinpath(filename))
            ).describe())

        ids = get_all_hexagon_ids(path_to_hexagon)

    else:
        for id in ids:
            data = get_hexagon_ts_by_id(id)
            descriptions_dfs.append(pd.DataFrame(data[0]).describe())

    joined = pd.concat(descriptions_dfs, axis=1)
    joined.columns = ids
    return joined.T




def get_passengers_data():
    print(os.getcwd())
    return pd.read_csv(get_data_dir_path().joinpath('misc/airline-passengers.csv'), sep=',', header=0)
    # return np.genfromtxt('../data/misc/airline-passengers.csv', delimiter=',')



class Dataset(Enum):
    LINREG = 1
    SINE = 2
    WIGGLY_SINE = 3
    PASSENGERS = 4
    HEXAGON = 5


def get_data(d=Dataset.LINREG, test_size=None, dim=1, hidx=None, scaler=MinMaxScaler(feature_range=(0, 1))):
    if scaler is None:
        scaler = FunctionTransformer(lambda x: x)  # identity

    n_train = None
    name = '.'
    anomaly_start = None
    anomaly_stop = None

    if d is Dataset.LINREG:
        data = datasets.gen_linear_regression_data(1000).values
        name = 'Linear Function'
    elif d is Dataset.SINE:
        data = datasets.gen_sine_wave()
        data = data[:, :dim]
        name = 'Sine'
    elif d is Dataset.WIGGLY_SINE:
        _, data = datasets.gen_noisy_wiggly_sine(1000)
        data = data.reshape(-1, 1)
        name = 'Wiggly Sine'
    elif d is Dataset.PASSENGERS:
        data = datasets.get_passengers()[['Passengers']]
    elif d is Dataset.HEXAGON:
        if hidx is None:
            raise ValueError("hix must be specified if Hexagon dataset in use")
        data, n_train, anomaly_start, anomaly_stop, original_id = get_hexagon_ts_by_id(hidx)
        name = f'Hexagon: {original_id}'
    else:
        data = datasets.gen_linear_regression_data(1000)
        name = 'Default: Linear Function'

    if test_size:
        n_train = int(data.shape[0] * (1 - test_size))
    elif d is not Dataset.HEXAGON:
        n_train = int(data.shape[0] * (1 - .2))
        # raise ValueError("If other than hexagon dataset is in use, test_size must be also specified")

    scaled = scaler.fit_transform(data)

    return data, scaled, n_train, anomaly_start, anomaly_stop, name



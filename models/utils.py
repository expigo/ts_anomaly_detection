import os
import random
from datetime import datetime
from pathlib import PurePath, Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_data_dir_path():
    return get_project_root().joinpath('data')

def print_labeled_locations() -> dict:
    """
    Curated list of "eye-labeled" datasets (source: hexagon)
    :return: dict[loc_id: str]: split_index:int
    """
    locs = {}
    locs['004'] = 5590
    locs['011'] = 11860
    locs['012'] = 16055
    return locs


def get_all_filenames(data_dir_path: str) -> dict:
    filenames = next(os.walk(data_dir_path), (None, None, []))[2]
    return {f.split(sep='_')[0]: (f, f.split('_')[3].split('.')[0]) for f in filenames}

def get_ts_by_id(id, data_dir_path='./data') -> (np.array, int):
    filenames = get_all_filenames(data_dir_path)
    file_path, split_index = filenames[f'{id}']
    return np.genfromtxt(PurePath(data_dir_path).joinpath(file_path), delimiter='\n'), int(split_index)

def plot_location_by_id(id, vlines = []):
    data, split_index = get_ts_by_id(id)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data)
    ax.vlines([split_index, *vlines], data.min(), data.max(), linestyles='dashed', colors='red')

    plt.title(f'location {id}\n split @ {split_index}')
    # fig.show()

def gen_sine_wave(N=100, L=1000, T=20):
    """
    :param N: no of samples
    :param L: length of each sample
    :param T: width of the wave
    :return:
    """

    x = np.empty((N, L), np.float32)
    x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)  # introduce some shifts
    y = np.sin(x / T).astype(np.float32)

    return y.T


def ts_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Works for multivariate data as well.

    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (to predict: [t,...,t+(n_out-1)] (y).
    dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:
    Pandas DataFrame of series framed for supervised learning.

    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('y%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('y%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('y%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def get_passengers():
    print(os.getcwd())
    return pd.read_csv(get_data_dir_path().joinpath('misc/airline-passengers.csv'), sep=',', header=0)
    # return np.genfromtxt('../data/misc/airline-passengers.csv', delimiter=',')

def sine_2(X, signal_freq=100.):
    # return np.sin(2 * np.pi * X / signal_freq)
    return (np.sin(2 * np.pi * X / signal_freq) + np.sin(4 * np.pi * X / signal_freq)) / 2.0


def noisy(Y, noise_range=(-.05, .05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return np.array(Y + noise)

def get_noisy_wiggly_sine(sample_size):
    random_offset = random.randint(0, sample_size)
    X = np.arange(sample_size)
    x_base = sine_2(X + random_offset)
    Y = noisy(x_base)
    return x_base, Y  # .reshape(-1, 1)

def get_linear_regression_data(n):
    X = np.linspace(-10, 10, num=n+1)
    y = 3 * X + 1
    start_time = datetime(2020, 7, 1, 1, 0, 0)
    timestamps = pd.Series(pd.date_range(
        start_time,
        freq='H',
        periods=y.shape[0]
    ))
    dataframe = pd.DataFrame(
        data=y,
        index=timestamps,
        columns=['data']
    )
    return dataframe

def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
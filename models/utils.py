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
    return np.genfromtxt(PurePath(path_to_hexagon).joinpath(file_path), delimiter='\n').reshape(-1, 1), \
           split_index, anomaly_start_idx, anomaly_stop_idx


def plot_hexagon_location_by_id(id, vlines=[]):
    data, split_index, anomaly_start_idx, anomaly_stop_idx = get_hexagon_ts_by_id(id)

    fig, ax = plt.subplots(figsize=(12, 6))
    # plt.figure(2)
    ax.plot(data, color='royalblue')
    ax.vlines([split_index, *vlines], *ax.get_ylim(), linestyles='dashed', colors='k', label='train/test split')
    ax.axvspan(anomaly_start_idx, anomaly_stop_idx, alpha=0.5, color='salmon', label='anomaly occurrence area')

    ax.set_title(f'location {id}\n split @ {split_index}')
    plt.legend()


def gen_sine_wave(N=100, L=1000, T=20):
    """
    :param N: no of samples
    :param L: length of each sample
    :param T: width of the wave
    :return:
    """

    x = np.empty((N, L), np.float32)
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)  # introduce some shifts
    y = np.sin(x / T).astype(np.float32)

    return y.T


def ts_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset. Create windows basically.
    Works for multivariate data as well.

    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X). (window size)
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
    X = np.linspace(-10, 10, num=n)
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


def save_model_old(model, history: dict, N_EPOCHS, d, n_layers, af, lr, hidx=None,
               name:str ='model_' + datetime.now().strftime("%Y%m%d-%H%M%S")):
    path = get_trained_models_dir_path().joinpath(name).joinpath('model')
    path.parent.mkdir(parents=True, exist_ok=True)
    desc_path = get_trained_models_dir_path().joinpath(name).joinpath(f'desc.txt')
    with open(desc_path, "w") as text_file:
        print(model, file=text_file)
        print(f'epochs: {N_EPOCHS}', file=text_file)
        print(f'datasetype: {d} ({hidx})', file=text_file)
        print(f'layers: {n_layers}', file=text_file)
        print(f'af: {af}', file=text_file)
        print(f'lr: {lr}', file=text_file)

    pd.DataFrame.from_dict(history).to_csv(path.parent.joinpath('loss_history.csv'), sep=',', index=False)

    torch.save(model, path)


def save_model(model, history: dict, model_params, training_params, d=None, hidx=None,
               name:str ='model_' + datetime.now().strftime("%Y%m%d-%H%M%S"), val_set:bool=True):
    root = get_trained_models_dir_path()
    path = root.joinpath(name).joinpath('model')
    path.parent.mkdir(parents=True, exist_ok=True)
    desc_path = root.joinpath(name).joinpath(f'desc.txt')
    with open(desc_path, "w") as text_file:
        print(model, file=text_file)
        print(f"epochs: {training_params['n_epochs']}", file=text_file)
        print(f'datasetype: {d} ({hidx})', file=text_file)
        print(f"layers: {model_params['n_layers']}", file=text_file)
        print(f"af: {model_params['af']}", file=text_file)
        print(f"lr: {training_params['lr']}", file=text_file)

    registry_path = root.joinpath('registry.csv')

    data_params = {
        "dataset": d,
        "hexagon_id": hidx,
        "val_set_used": val_set
    }
    pd.DataFrame(model_params | training_params | data_params, index=[name]).to_csv(registry_path,
                                                                      mode='a+',
                                                                      header=not os.path.exists(registry_path),
                                                                      index=[0])


    pd.DataFrame.from_dict(history).to_csv(path.parent.joinpath('loss_history.csv'), sep=',', index=False)


    torch.save(model, path)


def load_model(name=None):
    all_models_path = get_trained_models_dir_path()
    model_dirs = get_all_dirnames(all_models_path)
    if not model_dirs:
        raise ValueError('No saved models to choose from!')
    if name is None:
        # get last
        dates_sorted_desc = sorted(model_dirs,
                                   key=lambda d: datetime.strptime(d.split('_')[1], "%Y%m%d-%H%M%S"),
                                   reverse=True)
        last = Path(dates_sorted_desc[0]).joinpath('model')
        name = all_models_path.joinpath(last)
    else:
        name = all_models_path.joinpath(name).joinpath('model')

    trained = torch.load(name)
    return trained

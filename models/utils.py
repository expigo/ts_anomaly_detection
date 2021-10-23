import os
import random
from datetime import datetime
from pathlib import PurePath, Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

from models.simple_rnn.rnn import SimpleRNN


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


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)


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


    torch.save(model.state_dict(), path)


def load_model(model=None, name=None):
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

    if model is None:
        model = get_model_by_name(name.parent.stem)

    trained_dict = torch.load(name)
    model.load_state_dict(trained_dict)
    return model


def get_model_by_name(name):
    root = get_trained_models_dir_path()
    registry_path = root.joinpath('registry.csv')

    registry = pd.read_csv(registry_path, index_col=0)

    model_params = registry.loc[[name],
                     ['input_size', 'output_size', 'hidden_dim', 'n_layers', 'dropout', 'af']
    ].to_numpy().ravel()

    rnn = SimpleRNN(*model_params)
    return rnn



def test_train_split_by_idx(df, idx):
    pass


def feature_label_split(df, feature_size, values_only=False):
    X = df.iloc[:, :-feature_size]
    y = df.iloc[:, -feature_size:]
    if values_only:
        return X.values, y.values
    else:
        return X, y


def get_dataloaders(data, batch_size, input_size, output_size, val_set=True, n_train=None, test_size=0.2):
    """
    Splits data into train and test sets.
    Prepares DataLoaders for both.

    Args:
        batch_size:
        input_size: window size
        data:

    Returns:
    Create data loader and the index of dataset split
    """

    # create windows and prepare for batching
    labeled = ts_to_supervised(data, input_size, output_size, dropnan=True)
    dataset_size = labeled.shape[0]

    # ----- split into train and test sets -----

    if n_train is not None:
        X, y = feature_label_split(labeled, output_size, values_only=True)
        train_X, test_X = X[:n_train, :], X[n_train:, :]
        train_y, test_y = y[:n_train, :], y[n_train:, :]
        if val_set:
            test_dataset_size = test_X.shape[0]
            train_dataset_size = train_X.shape[0]
            test_ratio = test_dataset_size / dataset_size
            test_ratio = min(test_ratio, 0.2)
            val_ratio = test_ratio / (1 - test_ratio)
            n_val = int(val_ratio * train_dataset_size)
            train_X, train_y = train_X[:-n_val, :], train_y[:-n_val, :]
            val_X, val_y = train_X[-n_val:, :], train_y[-n_val:, :]
    else:
        X, y = feature_label_split(labeled, output_size)
        val_ratio = test_size / (1 - test_size)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=False)
        if val_set:
            train_X, val_X, test_y, val_y = train_test_split(train_X, train_y, test_size=val_ratio, shuffle=False)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # input shape: [batch_size, seq_length, features]
    x_train = torch.tensor(train_X, dtype=torch.float).unsqueeze(1)
    y_train = torch.tensor(train_y, dtype=torch.float).unsqueeze(1)
    x_test = torch.tensor(test_X, dtype=torch.float).unsqueeze(1)
    y_test = torch.tensor(test_y, dtype=torch.float).unsqueeze(1)
    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=False,
                                               worker_init_fn=random.seed(42),
                                               drop_last=True
                                               )

    train_loader_one = torch.utils.data.DataLoader(train,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=False,
                                                   worker_init_fn=random.seed(42),
                                                   drop_last=True
                                                   )


    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=False,
                                              worker_init_fn=random.seed(42),
                                              drop_last=True

                                              )

    test_loader_one = torch.utils.data.DataLoader(test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=False,
                                                  worker_init_fn=random.seed(42),
                                                  drop_last=True
                                                  )

    if val_set:
        x_val = torch.Tensor(val_X).unsqueeze(1)
        y_val = torch.Tensor(val_y).unsqueeze(1)

        val = torch.utils.data.TensorDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(val,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=False,
                                                 worker_init_fn=random.seed(42),
                                                 drop_last=True
                                                 )
        return train_loader, train_loader_one, test_loader, test_loader_one, val_loader, train_X
    # -------------------------------------------------------

    return train_loader, train_loader_one, test_loader, test_loader_one, None, train_X

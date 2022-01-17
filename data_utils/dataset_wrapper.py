from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from collections import namedtuple

import data_utils.utils as dutils
import models.utils as mutils

mutils.set_seed(42)


class TimeSeriesDataset:
    def __init__(self, data: pd.DataFrame,
                 name, n_train=None,
                 scaler='identity',
                 anomaly_start=None,
                 anomaly_stop=None):

        self.scaler = mutils.get_scaler(scaler)
        self.data = self.scaler.fit_transform(data)
        self.name = name
        self.n_train = n_train
        self.anomaly_start = anomaly_start
        self.anomaly_stop = anomaly_stop

    def get_scaled(self, scaler_name:str = None):
        return TimeSeriesDataset(data=self.data, name=self.name, n_train=self.n_train,
                                 anomaly_start=self.anomaly_start, anomaly_stop=self.anomaly_stop,
                                 scaler=scaler_name)

    @staticmethod
    def from_enum(d=dutils.Dataset.LINREG, hidx=None, scaler='identity'):
        if d is dutils.Dataset.HEXAGON and hidx is None:
            raise ValueError("ID needed for hexagon dataset!")
        original, _, n_train, anomaly_start, anomaly_stop, name = dutils.get_data(d, hidx=hidx)
        description = {
            "dataset_type": d,
            "hexagon_id": hidx,
            "scaler": scaler
        }
        return TimeSeriesDataset(data=pd.DataFrame(original), name=name, n_train=n_train,
                                 anomaly_start=anomaly_start, anomaly_stop=anomaly_stop, scaler=scaler), description

    def gen_time_lags(self, window_size, out_dim):
        return mutils.ts_to_supervised(self.data, window_size, out_dim)

    def test_train_split_by_index(self, n_train: int = None, feature_size=10, n_out=1, val_set=True):
        if not (n_train or self.n_train):
            raise ValueError("No index provided!")

        idx = n_train if n_train is not None else self.n_train
        labeled = self.gen_time_lags(feature_size, n_out)

        return index_test_train_split(idx, labeled, n_out, val_set)

    def test_train_split(self, test_size, feature_size=10, n_out=1, val_set=True):
        labeled = self.gen_time_lags(feature_size, n_out)

        return ratio_train_test_split(labeled, n_out, test_size, val_set)

    def test_train_split_by_index_from_labeled(self, labeled_data, n_out, n_train: int = None, val_set=True):
        if not (n_train or self.n_train):
            raise ValueError("No index provided!")

        idx = n_train if n_train is not None else self.n_train

        return index_test_train_split(idx, labeled_data, n_out, val_set)

    def get_dataloaders(self, batch_size, input_size, output_size=1, val_set=True, test_size: float = None,
                        create_labels=True):

        batch_size = int(batch_size)
        input_size = int(input_size)
        output_size = int(output_size)

        if create_labels:
            data = self.to_supervised(input_size, output_size, dropna=True)
        else:
            data = self.data

        if test_size is not None:
            splits = ratio_train_test_split(data, output_size, test_size, val_set)
        else:
            splits = index_test_train_split(idx=self.n_train, labeled=data, n_out=output_size, val_set=val_set)

            x_train = torch.tensor(splits.train_X, dtype=torch.float).unsqueeze(1)
            y_train = torch.tensor(splits.train_y, dtype=torch.float).unsqueeze(1)
            x_test = torch.tensor(splits.test_X, dtype=torch.float).unsqueeze(1)
            y_test = torch.tensor(splits.test_y, dtype=torch.float).unsqueeze(1)
            train = torch.utils.data.TensorDataset(x_train, y_train)
            test = torch.utils.data.TensorDataset(x_test, y_test)

            dataloaders = namedtuple('dataloaders',
                                     ['train_loader', 'train_loader_one',
                                      'test_loader', 'test_loader_one',
                                      'val_loader', 'val_loader_one']
                                     )

            train_loader = torch.utils.data.DataLoader(train,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=False,
                                                       worker_init_fn=worker_init_fn,
                                                       drop_last=True
                                                       )

            train_loader_one = torch.utils.data.DataLoader(train,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0,
                                                           pin_memory=False,
                                                           worker_init_fn=worker_init_fn,
                                                           drop_last=True
                                                           )

            test_loader = torch.utils.data.DataLoader(test,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      pin_memory=False,
                                                      worker_init_fn=worker_init_fn,
                                                      drop_last=True

                                                      )

            test_loader_one = torch.utils.data.DataLoader(test,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=0,
                                                          pin_memory=False,
                                                          worker_init_fn=worker_init_fn,
                                                          drop_last=True
                                                          )

            if val_set:
                x_val = torch.Tensor(splits.val_X).unsqueeze(1)
                y_val = torch.Tensor(splits.val_y).unsqueeze(1)

                val = torch.utils.data.TensorDataset(x_val, y_val)
                val_loader = torch.utils.data.DataLoader(val,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=0,
                                                         pin_memory=False,
                                                         worker_init_fn=worker_init_fn,
                                                         drop_last=True
                                                         )

                val_loader_one = torch.utils.data.DataLoader(val,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=0,
                                                             pin_memory=False,
                                                             worker_init_fn=worker_init_fn,
                                                             drop_last=True
                                                             )
                return dataloaders(train_loader, train_loader_one,
                                   test_loader, test_loader_one,
                                   val_loader, val_loader_one)

            return dataloaders(train_loader, train_loader_one,
                               test_loader, test_loader_one,
                               None, None)

    @staticmethod
    def test_train_split_from_labeled(labeled_data, n_out, test_size, val_set=True):
        return ratio_train_test_split(labeled_data, n_out, test_size, val_set)

    def plot(self, vlines=[]):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data, color='royalblue')
        ax.vlines([self.n_train, *vlines], *ax.get_ylim(), linestyles='dashed', colors='k', label='train/test split')
        ax.axvspan(self.anomaly_start, self.anomaly_stop, alpha=0.5, color='salmon', label='anomaly occurrence area')

        ax.set_title(f'{self.name}\n split @ {self.n_train}')
        plt.legend()

    def to_supervised(self, input_size, output_size, dropna=True):
        return mutils.ts_to_supervised(self.data, input_size, output_size, dropna)


def index_test_train_split(idx, labeled, n_out, val_set):
    X, y = mutils.feature_label_split(labeled, n_out, values_only=True)
    train_X, test_X = X[:idx, :], X[idx:, :]
    train_y, test_y = y[:idx, :], y[idx:, :]
    splits = namedtuple('splits', ['train_X', 'test_X', 'val_X', 'train_y', 'test_y', 'val_y'])
    if val_set:
        dataset_size = labeled.shape[0]
        test_dataset_size = test_X.shape[0]
        train_dataset_size = train_X.shape[0]
        test_ratio = test_dataset_size / dataset_size
        test_ratio = min(test_ratio, 0.2)
        val_ratio = test_ratio / (1 - test_ratio)
        n_val = int(val_ratio * train_dataset_size)
        train_X, train_y = train_X[:-n_val, :], train_y[:-n_val, :]
        val_X, val_y = train_X[-n_val:, :], train_y[-n_val:, :]
        return splits(
            train_X, test_X, val_X,
            train_y, test_y, val_y
        )
    return splits(
        train_X, test_X, None,
        train_y, test_y, None
    )


def ratio_train_test_split(labeled, n_out, test_size, val_set):
    X, y = mutils.feature_label_split(labeled, n_out)
    val_ratio = test_size / (1 - test_size)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=False)

    splits = namedtuple('splits', ['train_X', 'test_X', 'val_X', 'train_y', 'test_y', 'val_y'])

    if val_set:
        train_X, val_X, test_y, val_y = train_test_split(train_X, train_y, test_size=val_ratio, shuffle=False)
        return splits(
            train_X, test_X, val_X,
            train_y, train_y, val_y
        )
    return splits(
        train_X, test_X, None,
        train_y, train_y, None
    )


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

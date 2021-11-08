from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd

import data_utils.utils as dutils
import models.utils as mutils


class TimeSeriesDataset:
    def __init__(self, data: pd.DataFrame, name, n_train=None, anomaly_start=None, anomaly_stop=None):
        self.data = data
        self.name = name
        self.n_train = n_train
        self.anomaly_start = anomaly_start
        self.anomaly_stop = anomaly_stop

    def get_scaled(self, scaler):
        return scaler.fit_transform(self.data)

    @staticmethod
    def from_enum(d=dutils.Dataset.LINREG, hidx=None):
        if d is dutils.Dataset.HEXAGON and hidx is None:
            raise ValueError("ID needed for hexagon dataset!")
        original, _, n_train, anomaly_start, anomaly_stop, name = dutils.get_data(d, hidx=hidx)
        return TimeSeriesDataset(pd.DataFrame(original), name, n_train, anomaly_start, anomaly_stop)

    def gen_time_lags(self, window_size, out_dim):
        return mutils.ts_to_supervised(self.data, window_size, out_dim)

    def test_train_split_by_index(self, n_train:int=None, feature_size=10, n_out=1, val_set=True):
        if not (n_train or self.n_train):
            raise ValueError("No index provided!")

        idx = n_train if n_train is not None else self.n_train
        labeled = self.gen_time_lags(feature_size, n_out)

        return index_test_train_split(idx, labeled, n_out, val_set)

    def test_train_split(self, test_size, feature_size=10, n_out=1, val_set=True):
        labeled = self.gen_time_lags(feature_size, n_out)

        return ratio_train_test_split(labeled, n_out, test_size, val_set)

    def test_train_split_by_index_from_labeled(self, labeled_data, n_out, n_train: int=None, val_set=True):
        if not (n_train or self.n_train):
            raise ValueError("No index provided!")

        idx = n_train if n_train is not None else self.n_train

        return index_test_train_split(idx, labeled_data, n_out, val_set)

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


def index_test_train_split(idx, labeled, n_out, val_set):
    X, y = mutils.feature_label_split(labeled, n_out, values_only=True)
    train_X, test_X = X[:idx, :], X[idx:, :]
    train_y, test_y = y[:idx, :], y[idx:, :]
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
        return train_X, test_X, val_X, train_y, test_y, val_y
    return train_X, test_X, train_y, test_y


def ratio_train_test_split(labeled, n_out, test_size, val_set):
    X, y = mutils.feature_label_split(labeled, n_out)
    val_ratio = test_size / (1 - test_size)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=False)
    if val_set:
        train_X, val_X, test_y, val_y = train_test_split(train_X, train_y, test_size=val_ratio, shuffle=False)
        return train_X, test_X, val_X, train_y, test_y, val_y
    return train_X, test_X, train_y, test_y


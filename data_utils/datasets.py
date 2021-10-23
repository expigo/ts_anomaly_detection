import numpy as np
import pandas as pd
from pathlib import PurePath
from datetime import datetime
import random
import os

import data_utils.utils as utils

# ---------------- synthetic --------------------

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

def sine_2(X, signal_freq=100.):
    # return np.sin(2 * np.pi * X / signal_freq)
    return (np.sin(2 * np.pi * X / signal_freq) + np.sin(4 * np.pi * X / signal_freq)) / 2.0


def _noisify(Y, noise_range=(-.05, .05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return np.array(Y + noise)


def gen_noisy_wiggly_sine(sample_size):
    random_offset = random.randint(0, sample_size)
    X = np.arange(sample_size)
    x_base = sine_2(X + random_offset)
    Y = _noisify(x_base)
    return x_base, Y  # .reshape(-1, 1)


def gen_linear_regression_data(n):
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


# ---------------- /synthetic --------------------


def get_passengers():
    return utils.get_passengers_data()

def get_hexagon(id:int=4):
    return utils.get_hexagon_ts_by_id(id)
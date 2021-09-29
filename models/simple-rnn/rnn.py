from os import terminal_size

import torch
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error

import os, random, multiprocessing
from enum import Enum

from models.utils import gen_sine_wave, ts_to_supervised, get_passengers, \
    get_linear_regression_data, get_noisy_wiggly_sine, set_seed

set_seed(42)


class Dataset(Enum):
    LINREG = 1
    SINE = 2
    WIGGLY_SINE = 3
    PASSENGERS = 4


class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, active_fun='relu'):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, nonlinearity=active_fun, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.contiguous().view(batch_size, -1)
        output = self.fc(r_out)
        return output, hidden


def train_and_evaluate(n_epochs, lr, batch_size, hidden_dim, input_size, output_size, n_layers, dataset,
                       do_plot=True, activ_fun='relu'):

    original, scaled, scaler = get_data(dataset)
    train_loader, test_loader, n_train = get_dataloaders(batch_size, input_size, output_size, scaled)

    rnn = SimpleRNN(input_size, output_size, hidden_dim, n_layers, activ_fun)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    hidden = None  # initial hidden
    train_loss = []

    for epoch in range(n_epochs):

        for data, target in train_loader:

            if data.shape[0] != batch_size:
                print(f'Batch Size Validation- Input shape Issue: {data.shape}')
                continue
            else:
                optimizer.zero_grad()
                output, hidden = rnn(data, hidden)
                hidden = hidden.data
                loss = criterion(output, target)  # squeeze?
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        print(f"Epoch: {epoch}  | Average training Loss: {np.mean(train_loss)}")

    rmse, inv_y, inv_yhat = evaluate(rnn, scaler, test_loader)
    if do_plot:
        plot(original, dataset, hidden_dim, inv_y, inv_yhat, lr, n_train,
             rmse, train_loss, input_size, n_epochs, batch_size)

    return rmse


def evaluate(rnn, scaler, test_loader):
    # Prediction using tensor of predictors i.e x_test
    x_test = test_loader.dataset.tensors[0]
    yhat, _ = rnn(x_test, None)

    yhat = yhat.detach().numpy()

    # invert scaling for forecast
    test_X = x_test.numpy().reshape(-1, x_test.shape[2])
    inv_yhat = np.concatenate((yhat, test_X), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_loader.dataset.tensors[1].numpy()
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    from math import sqrt
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    return rmse, inv_y, inv_yhat


def plot(data_n, dataset, hidden_dim, inv_y, inv_yhat, lr, n_train, rmse, train_loss, input_size, n_epochs, batch_size):
    plt.plot(data_n)
    X = np.arange(start=n_train + input_size, stop=data_n.shape[0])
    plt.plot(X, inv_yhat, color='r', label='inv_yhat')

    for i, t in enumerate(X):
        plt.plot([t, t], [inv_yhat[i], inv_y[i]], 'k--', linewidth=.3, alpha=.5)
    plt.title(f"{dataset.name} |total test rmse: {rmse:.4f} | n_epochs: {n_epochs} | lr={lr}"
              f" | batch_size: {batch_size} | hidden_dim={hidden_dim}")
    plt.legend()
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(train_loss)
    plt.title("Learning curve")
    plt.show()
    print('Test RMSE: %.3f' % rmse)


def get_dataloaders(batch_size, input_size, output_size, scaled_data):
    """
    Splits data into train and test sets.
    Prepares DataLoaders for both.

    Args:
        batch_size:
        input_size: window size
        scaled_data:

    Returns:
    Create data loader and the index of dataset split
    """

    # prepare data for creating windows and batching
    n_out = 1  # one value ahead
    labeled = ts_to_supervised(scaled_data, input_size, output_size, dropnan=True)

    # ----- split into train and test sets -----
    values = labeled.values
    TRAIN_PERCENTAGE = .75
    n_train = int(TRAIN_PERCENTAGE * len(values))
    # n_train = 750
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    # test_X, test_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # input shape: [batch_size, seq_length, features]
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    x_train = torch.tensor(train_X, dtype=torch.float).unsqueeze(1)
    y_train = torch.tensor(train_y, dtype=torch.float).unsqueeze(1)
    x_test = torch.tensor(test_X, dtype=torch.float).unsqueeze(1)
    y_test = torch.tensor(test_y, dtype=torch.float).unsqueeze(1)
    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               worker_init_fn=random.seed(42)
                                               )
    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=True,
                                              worker_init_fn=random.seed(42)
                                              )
    # -------------------------------------------------------

    return train_loader, test_loader, n_train


def get_data(d: Dataset = Dataset.LINREG, dim=1, scaler=MinMaxScaler(feature_range=(0, 1))):
    if d is Dataset.LINREG:
        data = get_linear_regression_data(1000).values
    elif d is Dataset.SINE:
        data = gen_sine_wave()
        data = data[:, :dim]
    elif d is Dataset.WIGGLY_SINE:
        _, data = get_noisy_wiggly_sine(1000)
        data = data.reshape(-1, 1)
    elif d is Dataset.PASSENGERS:
        data = get_passengers()[['Passengers']]
    else:
        data = get_linear_regression_data(1000).values

    # scaler = FunctionTransformer(lambda x: x)
    scaled = scaler.fit_transform(data)
    return data, scaled, scaler


def reproduce_best(d: Dataset):
    if d is Dataset.LINREG:
        N_EPOCHS = 100
        lr = 0.01
        batch_size = 2
        hidden_dim = 10
        n_layers = 1
        af = 'relu'

        rmse = train_and_evaluate(N_EPOCHS, lr, batch_size, hidden_dim, n_layers, 1, 1,
                                  d, activ_fun=af, do_plot=True)
    elif d is Dataset.SINE:
        N_EPOCHS = 100
        lr = 0.01
        batch_size = 2
        hidden_dim = 10
        n_layers = 1
        af = 'relu'

        rmse = train_and_evaluate(N_EPOCHS, lr, batch_size, hidden_dim, n_layers, 1, 1,
                                  d, activ_fun=af, do_plot=True)
    elif d is Dataset.WIGGLY_SINE:
        N_EPOCHS = 100
        lr = 0.01
        batch_size = 2
        hidden_dim = 10
        n_layers = 1
        af = 'tanh'

        rmse = train_and_evaluate(N_EPOCHS, lr, batch_size, hidden_dim, n_layers, 1, 1,
                                  d, activ_fun=af, do_plot=True)
    elif d is Dataset.PASSENGERS:
        N_EPOCHS = 20
        lr = 0.01
        batch_size = 2
        hidden_dim = 10
        n_layers = 1
        af = 'tanh'

        rmse = train_and_evaluate(N_EPOCHS, lr, batch_size, hidden_dim, n_layers, 1, 1,
                                  d, activ_fun=af, do_plot=True)
    else:
        rmse = 0

    return rmse
# reproduce_best(Dataset.PASSENGERS)


def calc_stats(r: list):
    return np.array([np.mean(r), np.std(r)])


cpu_count = multiprocessing.cpu_count()


def calc(i, hp, dataset):
    hp = hp[1]
    print(i)
    curr_set = Parallel(n_jobs=cpu_count)(delayed(test)(i, j, hp, dataset) for j in range(10))
    stats = calc_stats(curr_set)
    # print(stats)
    # print(curr_set)
    path = f".\\results\\tanh\\{dataset.name}"
    os.makedirs(path, exist_ok=True)  # !!!
    np.savetxt(f".\\results\\tanh\\{dataset.name}\\{hp}_results.csv", curr_set, delimiter=",")
    np.savetxt(f".\\results\\tanh\\{dataset.name}\\{hp}_stats.csv", stats, delimiter=",")
    return hp, curr_set


def test(i, j, hp, dataset):
    print(f'{i}.{j}')
    rmse = train_and_evaluate(hp[0], hp[1], hp[2], hp[3], hp[4], 1, 1, dataset,
                              do_plot=False)
    return rmse


def multi_eval(test_cases):
    fn = lambda d: Parallel(n_jobs=cpu_count)(delayed(calc)(i, hp, d) for i, hp in enumerate(test_cases.items()))

    Parallel(n_jobs=cpu_count)(delayed(fn)(d) for d in
                               (Dataset.LINREG, Dataset.SINE, Dataset.WIGGLY_SINE, Dataset.PASSENGERS))

    for d in (Dataset.SINE, Dataset.LINREG, Dataset.WIGGLY_SINE, Dataset.PASSENGERS):
        print(d)
        res = Parallel(n_jobs=4)(delayed(calc)(i, hp, Dataset.LINREG) for i, hp in enumerate(test_cases.items()))

    print(f'>>> {res}')



if __name__ == "__main__":

    # ------------- single --------------

    input_size = 1  # window size
    output_size = 1  # TODO

    N_EPOCHS = 10
    lr = 0.01
    batch_size = 2
    hidden_dim = 10
    n_layers = 1
    af = 'tanh'

    rmse = train_and_evaluate(N_EPOCHS, lr, batch_size, hidden_dim, input_size, output_size, n_layers,
                              Dataset.WIGGLY_SINE, activ_fun=af, do_plot=True)

    # ------------ /single ---------------

    # # ---- multi hyperparameter sets evaluation (set do_plot=False)
    # # i: (N_EPOCHS, lr, batch_size, hidden_dim, n_layers)
    # test_cases = {0: (10, 0.01, 2, 10, 1),
    #           1: (10, 0.001, 2, 10, 1),
    #           2: (10, 0.0001, 2, 10, 1),
    #           3: (20, 0.01, 2, 10, 1),
    #           4: (50, 0.01, 2, 10, 1),
    #           5: (100, 0.01, 2, 10, 1),
    #           6: (20, 0.01, 5, 10, 1),
    #           7: (20, 0.01, 10, 10, 1),
    #           8: (20, 0.01, 20, 16, 1),
    #           9: (20, 0.01, 10, 64, 1),
    #           10: (20, 0.01, 10, 256, 1),
    #           11: (50, 0.001, 50, 128, 1),
    #           # 11: (20, 0.01, 10, 1024, 1)     # TODO check for exploding gradient
    #           }
    # multi_eval(test_cases)
    # # ------ /multi ------------

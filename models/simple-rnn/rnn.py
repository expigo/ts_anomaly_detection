import copy
import random
from enum import Enum

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, MaxAbsScaler, RobustScaler
import seaborn as sns
from scipy import stats

from models.utils import gen_sine_wave, ts_to_supervised, get_passengers, \
    get_linear_regression_data, get_noisy_wiggly_sine, set_seed, get_hexagon_ts_by_id, plot_hexagon_location_by_id, \
    save_model, load_model

set_seed(42)
plt.style.use('ggplot')


class Dataset(Enum):
    LINREG = 1
    SINE = 2
    WIGGLY_SINE = 3
    PASSENGERS = 4
    HEXAGON = 5


class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0, af='relu'):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, nonlinearity=af, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        r_out, hidden = self.rnn(x, hidden)
        # batch_size = x.size(0)
        # r_out = r_out.contiguous().view(batch_size, -1)
        output = self.fc(r_out)
        return output, hidden


def get_data(d=Dataset.LINREG, test_size=None, dim=1, hidx=None, scaler=MinMaxScaler(feature_range=(0, 1))):
    if scaler is None:
        scaler = FunctionTransformer(lambda x: x)  # identity

    n_train = None
    anomaly_start = None
    anomaly_stop = None

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
    elif d is Dataset.HEXAGON:
        if hidx is None:
            raise ValueError("hix must be specified if Hexagon dataset in use")
        data, n_train, anomaly_start, anomaly_stop = get_hexagon_ts_by_id(hidx)
    else:
        data = get_linear_regression_data(1000)

    if test_size:
        n_train = int(data.shape[0] * (1 - test_size))
    elif d is not Dataset.HEXAGON:
        n_train = int(data.shape[0] * (1 - .2))
        # raise ValueError("If other than hexagon dataset is in use, test_size must be also specified")

    scaled = scaler.fit_transform(data)

    return data, scaled, n_train, anomaly_start, anomaly_stop


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


def train(model, n_epochs, train_loader, val_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    hidden = None  # initial hidden

    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(1, n_epochs+1):
        model = model.train()

        train_loss = []
        for data, target in train_loader:

            if data.shape[0] != batch_size:
                print(f'Batch Size Validation- Input shape Issue: {data.shape}'
                      f' [input size != batch size ({data.shape[0]}!={batch_size})]')
                continue
            else:
                optimizer.zero_grad()
                output, hidden = model(data, hidden)
                hidden = hidden.data
                loss = criterion(output, target)  # squeeze?
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        model = model.eval()
        val_loss = []
        if val_loader is not None:
            with torch.no_grad():
                for data, target in val_loader:
                    if data.shape[0] != batch_size:
                        print(f'Batch Size Validation- Input shape Issue: {data.shape}'
                              f' [input size != batch size ({data.shape[0]}!={batch_size})]')
                        continue
                    else:
                        output, hidden = model(data, hidden)
                        hidden = hidden.data
                        loss = criterion(output, target)
                        val_loss.append(loss.item())

        history['train'].append(np.mean(train_loss))

        if val_loader is not None:
            val_loss = np.mean(val_loss)
            history['val'].append(np.mean(val_loss))
            val_msg = f" |  Avg val loss {np.mean(val_loss)}"
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            val_msg = ""
            history['val'].append(np.nan)




        msg = f"Epoch: [{epoch}/{n_epochs}]  |" \
                f" Average training Loss: {np.mean(train_loss)} {val_msg}"
        print(msg)


    if val_loader is not None:
        model.load_state_dict(best_model_wts)

    return model.eval(), history

def get_model(model, model_params):
    models = {
        "rnn": SimpleRNN,
    }
    return models.get(model.lower())(**model_params)


def get_scaler(scaler):
    scalers = {
        "minmax": lambda: MinMaxScaler(feature_range=(0 ,1)),
        "identity": lambda: FunctionTransformer(lambda x: x),
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


def evaluate(model, test_loader, scaler=get_scaler('identity'), criterion=lambda:torch.nn.L1Loss(reduction=sum)):
    predictions = []
    values = []

    model = model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            y_hat, _ = model(x, None)
            # TODO: the code below most probably works fine only for out_dim=1: closer investigation required
            y_hat = y_hat.detach().numpy().reshape(-1, 1)
            y = y.detach().numpy().reshape(-1, 1)

            predictions.append(y_hat)
            values.append(y)

    return np.array(values).reshape(-1, 1), np.array(predictions).reshape(-1, 1)


def plot(dataset, test_idx, y_test, y_hat, title="result"):
    X = np.arange(test_idx, len(y_test)+test_idx)
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.scatter(x=X, y=y_test, label='y')
    # ax.scatter(x=X, y=y_hat, label='y_hat')
    ax.plot(X, y_test, label='y')
    ax.plot(X, y_hat, label='y_hat')
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    # plt.show()


def show_table_params():
    global table
    table, ax = plt.subplots()
    plt.figure(3)
    table.set_size_inches(8, 2)
    # hide axes
    table.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(np.column_stack([input_size, output_size, N_EPOCHS, lr, batch_size, hidden_dim, n_layers, af]),
                      columns=['in_size', 'out_size', 'n_epochs', 'lr', 'batch_size', 'hidden_dim', 'n_layers',
                               'act. fn.'])
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.tight_layout()
    # plt.show()


def reconstruct_ts(model, dataloder):

    # trained = model.eval()
    # y_train_one, _ = trained(train_loader_one.dataset.tensors[0], None)
    # train_mae_one = np.abs(train_loader_one.dataset.tensors[1].detach().numpy().ravel() -
    #                        y_train_one.detach().numpy().ravel())

    predictions = []
    criterion = torch.nn.L1Loss(reduction='sum')
    losses = []
    for x, y in dataloder:
        model = model.eval()
        yhat, _ = model(x, None)

        losses.append(criterion(y, yhat).item())
        predictions.append(yhat.detach().numpy().ravel())

    return np.array(predictions), np.array(losses)


def get_threshold(losses:np.array) -> float:
    bins = np.linspace(losses.min(), losses.max(), 50)
    bin_nums = np.digitize(losses, bins) - 1
    hist_vals = np.bincount(bin_nums)
    # plt.figure(3)
    # plt.title('y_train')
    # plt.plot(original, label="original")
    #
    # train_vals = y_train.detach().numpy().reshape(-1, 1)
    # plt.plot(scaler.inverse_transform(train_vals), label='y_train')
    # plt.plot(scaler.inverse_transform(train_X), label='train_x')
    # plt.legend()
    # plt.plot(scaler.inverse_transform(np.array(yy).reshape(-1, 1)), label='yy')
    # _, losses = reconstruct_ts(rnn, train_X)
    fg = sns.displot(losses, bins=50, kde=True)

    return losses.max()

if __name__ == "__main__":
    # ------------- single --------------

    input_size = 7  # window size
    output_size = 1

    N_EPOCHS = 100

    lr = 0.001
    batch_size = 4
    hidden_dim = 10
    n_layers = 3
    af = 'relu'
    dropout = 0

    model_params = {
                    'input_size': input_size,
                    'output_size': output_size,
                    'hidden_dim': hidden_dim,
                    'n_layers': n_layers,
                    'af': af,
                    'dropout': dropout
                    }

    training_params = {
        'n_epochs': N_EPOCHS,
        'lr': lr,
        'batch_size': batch_size
    }

    val_set = True

    d = Dataset.HEXAGON
    hidx = 129
    plot_hexagon_location_by_id(hidx)

    scaler = get_scaler('minmax')
    # show_table_params()

    # 1. get data
    original, scaled, n_train, anomaly_start, anomaly_stop = get_data(d, scaler=scaler, hidx=hidx)

    # 2. get dataloaders
    train_loader, train_loader_one, test_loader, test_loader_one, val_loader, train_X \
        = get_dataloaders(scaled, batch_size, input_size, output_size, n_train=n_train, val_set=val_set)

    # 3. get_model
    rnn = get_model('rnn', model_params)

    # 4. train
    # trained, history = train(rnn, N_EPOCHS, train_loader, val_loader=val_loader)
    # save_model(trained, history, model_params, training_params, d, hidx, val_set=val_set)

    # or use trained one
    trained = load_model('model_20211014-201349')  # exemplary model that works

    # 5. evaluate
    y_test, y_hat = evaluate(trained, test_loader_one, scaler)
    y_train, y_hat_train = evaluate(trained, train_loader_one, scaler)
    plot(original, 0, y_train, y_hat_train, title="training")

    y_test = scaler.inverse_transform(y_test)
    y_hat = scaler.inverse_transform(y_hat)

    mae = mean_absolute_error(y_test, y_hat)
    rmse = mean_squared_error(y_test, y_hat) ** .5
    r2 = r2_score(y_test, y_hat)
    print(mae, rmse, r2)

    plot(original, n_train, y_test, y_hat, title="test")
    # ----- anomaly detection ------
    reconstructed_train_data, losses = reconstruct_ts(trained, train_loader_one)
    print('max mae:', losses.max(), 'at', np.argwhere(losses == losses.max()))

    THRESHOLD = get_threshold(losses)

    reconstructed_test_data, losses_test = reconstruct_ts(trained, test_loader_one)
    anomalies_count = sum(l >= THRESHOLD for l in losses_test)
    anomalies_idxs = np.argwhere(losses_test >= THRESHOLD)
    print(anomalies_idxs + n_train)
    print(f'indicies should be in closed range: [{anomaly_start}, {anomaly_stop}]')
    print('done')
    plt.show()
    # ----- /anomaly detection ------


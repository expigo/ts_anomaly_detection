import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KernelDensity
import seaborn as sns

import models.utils as model_utils
import data_utils.utils as data_utils
from models.sensei import Sensei

model_utils.set_seed(42)
plt.style.use('ggplot')


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



def evaluate(model, test_loader, scaler=model_utils.get_scaler('identity'), criterion=lambda:torch.nn.L1Loss(reduction=sum)):
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


def get_threshold(losses: np.array, kde=False) -> float:
    bins = np.linspace(losses.min(), losses.max(), 50)
    bin_nums = np.digitize(losses, bins) - 1
    hist_vals = np.bincount(bin_nums)

    gt = losses
    if kde is not None:

        if isinstance(kde, float):
            kde = KernelDensity(bandwidth=.1).fit(losses[:, None])
        else:
            params = {'bandwidth': np.logspace(-1, 1, 100)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(losses[:, None])

            kde = grid.best_estimator_

        # losses = losses.reshape(-1, 1)
        scores = kde.score_samples(losses[:, None])
        sns.displot(scores, bins=50, kde=True)
        threshold = np.quantile(scores, .0000001)
        gt = scores
    else:
        kde = None
        threshold = losses.max()
        print('max mae:', threshold, 'at', np.argwhere(losses == losses.max()))


    return threshold, kde, gt


def detect_anomalies(model, train_data, test_data, kde=True):
    reconstructed_train_data, losses = reconstruct_ts(model, train_data)
    reconstructed_test_data, losses_test = reconstruct_ts(model, test_data)

    THRESHOLD, kde_test, train_scores = get_threshold(losses, kde=kde)

    if kde:
        losses_test = kde_test.score_samples(losses_test.reshape(-1, 1))

    anomalies_count = sum(l >= THRESHOLD for l in losses_test)
    print(f'Number of anomalies found: {anomalies_count}')
    anomalies_idxs = np.argwhere(losses_test <= THRESHOLD) + n_train
    return anomalies_idxs


if __name__ == "__main__":
    # ------------- single --------------

    input_size = 72  # window size
    output_size = 1

    N_EPOCHS = 10

    lr = 0.01
    batch_size = 32
    hidden_dim = 10
    n_layers = 1
    af = 'tanh'
    dropout = 0
    weight_decay = 1e-6


    model_params = {
                    'input_size': input_size,
                    'output_size': output_size,
                    'hidden_dim': hidden_dim,
                    'n_layers': n_layers,
                    'af': af,
                    'dropout': dropout,
                    }

    training_params = {
        'n_epochs': N_EPOCHS,
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay
    }

    val_set = True

    d = data_utils.Dataset.HEXAGON
    hidx = 4
    data_utils.plot_hexagon_location_by_id(hidx)
    # plt.show()

    scaler = model_utils.get_scaler('minmax')

    # 1. get data
    original, scaled, n_train, anomaly_start, anomaly_stop, name = data_utils.get_data(d, scaler=scaler, hidx=hidx)

    # 2. get dataloaders
    train_loader, train_loader_one, test_loader, test_loader_one, val_loader, train_X \
        = model_utils.get_dataloaders(scaled, batch_size, input_size, output_size, n_train=n_train, val_set=val_set)

    # 3. get_model
    model = model_utils.get_model('rnn', model_params)

    # 4. train
    # trained, history = train(rnn, N_EPOCHS, train_loader, val_loader=val_loader)
    # model_utils.save_model(trained, history, model_params, training_params, d, hidx, val_set=val_set)
    # or use trained one
    trained = model_utils.load_model()

    #
    # # 5. evaluate
    # y_test, y_hat = evaluate(trained, test_loader_one, scaler)
    # y_train, y_hat_train = evaluate(trained, train_loader_one, scaler)
    # plot(original, 0, y_train, y_hat_train, title="training")

    # # 4s.
    # loss_fn = torch.nn.MSELoss(reduction="mean")
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # s = Sensei(model, optimizer, loss_fn)
    #
    # # 5s.
    # trained, history = s.train(N_EPOCHS, train_loader, val_loader)
    # s.plot_losses()
    y_test, y_hat = s.evaluate(
        test_loader=test_loader_one
    )

    y_test = scaler.inverse_transform(y_test)
    y_hat = scaler.inverse_transform(y_hat)

    mae = mean_absolute_error(y_test, y_hat)
    rmse = mean_squared_error(y_test, y_hat) ** .5
    r2 = r2_score(y_test, y_hat)
    print(mae, rmse, r2)

    plot(original, n_train, y_test, y_hat, title="test")
    # ----- anomaly detection ------
    anomalies_idx = detect_anomalies(trained, train_loader_one, test_loader_one, kde=.1)
    print(anomalies_idx)
    print(f'indicies should be in closed range: [{anomaly_start}, {anomaly_stop}]')
    print('done')
    # ----- /anomaly detection ------
    plt.show()



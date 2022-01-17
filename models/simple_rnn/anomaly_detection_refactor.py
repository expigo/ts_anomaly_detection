import copy

import numpy as np
import torch
from ipython_genutils.py3compat import input
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KernelDensity
import seaborn as sns

import models.utils as model_utils
import data_utils.utils as data_utils
from models.sensei import Sensei

model_utils.set_seed(42)
plt.style.use('bmh')
# plt.style.use('seaborn-paper')


import matplotlib
matplotlib.use("TkAgg")
#

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


def plot(dataset, test_idx, y_test, y_hat, start, stop, anomalies_found, input_size, title="result"):
    X = np.arange(test_idx+input_size, len(y_test)+test_idx+input_size)
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.scatter(x=X, y=y_test, label='y')
    # ax.scatter(x=X, y=y_hat, label='y_hat')

    ax.axvspan(start, stop, alpha=0.2, color='salmon', label='expected anomaly occurrence area')
    # ax.scatter(x=anomalies_found, y=dataset[anomalies_found+input_size],
    #            marker='x', c='purple', label='anomalies found - input_sizex')
    # ax.plot(dataset[input_size:], label="dataset")
    # ax.plot(dataset, label="dataset")
    ax.plot(X, y_test, label='ground truth')
    ax.plot(X, y_hat, label='predictions')
    ax.scatter(x=anomalies_found+input_size, y=y_test[anomalies_found-test_idx],
               marker='x', c='black', label='anomalies detected')
    # ax.scatter(x=anomalies_found+input_size, y=dataset[anomalies_found],
    #            marker='x', c='black', label='anomalies detected')
    # ax.scatter(x=anomalies_found+input_size, y=y_test[anomalies_found-test_idx+input_size],
    #            marker='o', c='yellow', label='anomalies detected shifted')
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
    # criterion = torch.nn.L1Loss(reduction='sum')
    criterion = torch.nn.MSELoss(reduction='sum')
    losses = []
    for x, y in dataloder:
        model = model.eval()
        yhat, _ = model(x, None)

        losses.append(criterion(y, yhat).item())
        predictions.append(yhat.detach().numpy().ravel())

    return np.array(predictions), np.array(losses)


def get_threshold(losses: np.array, kde=False):
    # bins = np.linspace(losses.min(), losses.max(), 50)
    # bin_nums = np.digitize(losses, bins) - 1
    # hist_vals = np.bincount(bin_nums)

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
        # plt.close(); plt.plot(losses_test, np.zeros_like(losses_test) + 0., marker='x', ms=2); plt.plot(losses_test[anomalies_idxs.ravel() - n_train], np.zeros_like(losses_test[anomalies_idxs.ravel() - n_train]) + 0., 'x', color='black', ms=20); plt.show()


    anomalies_count = sum(l <= THRESHOLD for l in losses_test)
    print(f'Number of anomalies found: {anomalies_count}')
    l = losses_test[losses_test <= THRESHOLD]
    anomalies_idxs = np.argwhere(losses_test <= THRESHOLD) + n_train  # + input_size
    anomalies_with_scores = list(zip(l, anomalies_idxs.ravel()))
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(losses_test, np.zeros_like(losses_test) + 0., marker='o', ms=10, linestyle="None", label='all scores')
    ax.plot(losses_test[np.arange(anomaly_start - n_train, anomaly_stop - n_train)], np.zeros_like(losses_test[np.arange(anomaly_start - n_train, anomaly_stop - n_train)]) + 0., 'H', ms=6, label='GT')
    # ax.plot(losses_test[anomalies_idxs.ravel() - n_train], np.zeros_like(losses_test[anomalies_idxs.ravel() - n_train]) + 0., 'x', color='black', ms=10, label='Found Anomalies')
    plt.axvline(THRESHOLD, ls='--', c='black', ymin=0.4, ymax=0.6, label='threshold')
    ax.set_xlabel('log-likelihood of each sample under the model')
    ax.yaxis.set_visible(False)
    ax.grid(None)
    ax.set_ylim([-0.1, 0.1])
    plt.title(f'Scores and Foundings')
    plt.legend()
    plt.show()
    # TP = set.intersection(set(anomalies_idx.ravel()),set(np.arange(anomaly_start, anomaly_stop)))
    # TP = len(TP) / np.arange(anomaly_start, anomaly_stop).shape[0]
    #
    # FP = set.difference(set(anomalies_idx.ravel()),set(np.arange(anomaly_start, anomaly_stop)))
    # FP = len(FP)
    #
    # # TN =
    return anomalies_idxs


if __name__ == "__main__":
    # ------------- single --------------

    # input_size = 25  # window size
    # output_size = 1
    #
    # N_EPOCHS = 45
    #
    # lr = 0.08341537
    # batch_size = 36
    # hidden_dim = 23
    # n_layers = 2
    # af = 'tanh'
    # dropout = 0
    # weight_decay = 1e-6
    #
    # input_size = 33  # window size
    # output_size = 1
    #
    # N_EPOCHS = 40
    #
    # lr = 0.054969
    # batch_size = 50
    # hidden_dim = 10
    # n_layers = 4
    # af = 'tanh'
    # dropout = 0
    # weight_decay = 1e-6

    # input_size = 83  # window size
    # output_size = 1
    #
    # N_EPOCHS = 29
    #
    # lr = 0.020273
    # batch_size = 29
    # hidden_dim = 3
    # n_layers = 5
    # af = 'tanh'
    # dropout = 0
    # weight_decay = 1e-6

    # input_size = 100  # window size
    # output_size = 1
    #
    # N_EPOCHS = 12
    #
    # lr = 1.83015074e-02
    # batch_size = 27
    # hidden_dim = 10
    # n_layers = 4
    # af = 'tanh'
    # dropout = 0
    # weight_decay = 1e-6

    # input_size = 53  # window size
    # output_size = 1
    #
    # N_EPOCHS = 50
    #
    # lr = 1.61063959e-02
    # batch_size = 59
    # hidden_dim = 2
    # n_layers = 1
    # af = 'tanh'
    # dropout = 0
    # weight_decay = 1e-6

    # relu
    # input_size = 100  # window size
    # output_size = 1
    #
    # N_EPOCHS = 11
    #
    # lr = 3.94962999e-02
    # batch_size = 64
    # hidden_dim = 4
    # n_layers = 3
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # input_size = 53  # window size
    # output_size = 1
    #
    # N_EPOCHS = 1000
    #
    # lr = 2.29187947e-03
    # batch_size = 4
    # hidden_dim = 1
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    #
    # input_size = 68  # window size
    # output_size = 1
    #
    # N_EPOCHS = 50
    #
    # lr = 1.17195672e-03
    # batch_size = 8
    # hidden_dim = 1
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # gru # veru good!
    # input_size = 68  # window size
    # output_size = 1
    #
    # N_EPOCHS = 1000
    #
    # lr = 1.17195672e-03
    # batch_size = 8
    # hidden_dim = 1
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # gru  # even better!
    # # !!!! [4] MSE sum, gru
    # input_size = 83  # window size
    # output_size = 1
    #
    # N_EPOCHS = 50
    #
    # lr = 3.78994644e-03
    # batch_size = 4
    # hidden_dim = 1
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # rnn [34]
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 200
    #
    # lr = 9.03748016e-02
    # batch_size = 64
    # hidden_dim = 3
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # gru [34]
    # input_size = 61  # window size
    # output_size = 1
    #
    # N_EPOCHS = 200
    #
    # lr = 3.62116054e-02
    # batch_size = 53
    # hidden_dim = 6
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # [4] MSE sum, gru
    # input_size = 51  # window size
    # output_size = 1
    #
    # N_EPOCHS = 1000
    #
    # lr = 6.56958067e-02
    # batch_size = 18
    # hidden_dim = 5
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # [22] MSE sum, gru
    # input_size = 63  # window size
    # output_size = 1
    #
    # N_EPOCHS = 91
    #
    # lr = 1.36984689e-02
    # batch_size = 21
    # hidden_dim = 5
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # [122] MSE sum, gru
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 8.50249629e-02
    # batch_size = 116
    # hidden_dim = 9
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # [122] MSE sum, gru
    # input_size = 51  # window size
    # output_size = 1
    #
    # N_EPOCHS = 98
    #
    # lr = 4.81456266e-02
    # batch_size = 71
    # hidden_dim = 2
    # n_layers = 2
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # # [28] MSE sum, gru
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 7.76921265e-02
    # batch_size = 104
    # hidden_dim = 11
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # [27] MSE sum, gru (return rmse)
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 7.12488086e-02
    # batch_size = 98
    # hidden_dim = 13
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6


    # [122] MSE sum, rnn  !!! no anomalies found !!!
    # input_size = 63  # window size
    # output_size = 1
    #
    # N_EPOCHS = 99
    #
    # lr = 6.34079547e-02
    # batch_size = 78
    # hidden_dim = 9
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # !!!! [129] MSE sum, gru
    # input_size = 106  # window size
    # output_size = 1
    #
    # N_EPOCHS = 191
    #
    # lr = 1.09236731e-03
    # batch_size = 114
    # hidden_dim = 10
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # # !!!! [28] MSE sum, gru
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 171
    #
    # lr = 7.26141563e-02
    # batch_size = 94
    # hidden_dim = 13
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # # !!!! [129] MSE sum, gru
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 9.10361897e-02
    # batch_size = 123
    # hidden_dim = 5
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # # !!!! [129] MSE sum, lstm
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 9.03945573e-02
    # batch_size = 128
    # hidden_dim = 2
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # !!!! [129] MSE sum, rnn
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 7.96806693e-02
    # batch_size = 118
    # hidden_dim = 4
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # gru  # even better!
    # # !!!! [129] MSE sum, lstm
    # input_size = 59  # window size
    # output_size = 1
    #
    # N_EPOCHS = 91
    #
    # lr = 2.69828772e-02
    # batch_size = 126
    # hidden_dim = 18
    # n_layers = 2
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # # [129] lstm, no data scaling
    # input_size = 50  # window size
    # output_size = 1
    #
    # N_EPOCHS = 100
    #
    # lr = 7.717512296e-02
    # batch_size = 103
    # hidden_dim = 16
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6


    # # [28] lstm
    # input_size = 54  # window size
    # output_size = 1
    #
    # N_EPOCHS = 297
    #
    # lr = 4.35448198e-02
    # batch_size = 107
    # hidden_dim = 19
    # n_layers = 1
    # af = 'relu'
    # dropout = 0
    # weight_decay = 1e-6

    # TESTT lstm
    input_size = 15  # window size
    output_size = 1

    N_EPOCHS = 49

    lr = 0.0697368059322561
    batch_size = 58
    hidden_dim = 25
    n_layers = 21
    af = 'relu'
    dropout = 0
    weight_decay = 0

    model_params = {
                    'input_size': input_size,
                    'output_size': output_size,
                    'hidden_dim': hidden_dim,
                    'n_layers': n_layers,
                    # 'af': af,
                    'dropout_prob': dropout,
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
    train_loader, train_loader_one, test_loader, test_loader_one, val_loader, val_loader_one, train_X \
        = model_utils.get_dataloaders(scaled, batch_size, input_size, output_size, n_train=n_train, val_set=val_set)

    # 3. get_model
    model = model_utils.get_model('lstm', model_params)
    # rnn = model_utils.get_model('rnn', model_params)

    # # 4. train
    # trained, history = train(rnn, N_EPOCHS, train_loader, val_loader=val_loader)
    # model_utils.save_model(trained, history, model_params, training_params, d, hidx, val_set=val_set)
    # # or use trained one
    # # trained = model_utils.load_model()
    #
    # #
    # # # 5. evaluate
    # y_test, y_hat = evaluate(trained, test_loader_one, scaler)
    # y_train, y_hat_train = evaluate(trained, train_loader_one, scaler)
    # plot(original, 0, y_train, y_hat_train, title="training")

    # 4s.
    loss_fn = torch.nn.MSELoss(reduction="sum")
    # loss_fn = torch.nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    s = Sensei(model, optimizer, loss_fn)

    # 5s.
    trained, history = s.train(N_EPOCHS, train_loader, val_loader)

    # s, trained = Sensei.from_trained('model_20211022-152944')

    y_test, y_hat = s.evaluate(
        test_loader=test_loader_one
    )

    y_test = scaler.inverse_transform(y_test)
    y_hat = scaler.inverse_transform(y_hat)

    mae = mean_absolute_error(y_test, y_hat)
    rmse = mean_squared_error(y_test, y_hat) ** .5
    r2 = r2_score(y_test, y_hat)
    print(mae, rmse, r2)

    # ----- anomaly detection ------
    anomalies_idx = detect_anomalies(trained, train_loader_one, test_loader_one) #, kde=.1)
    print(anomalies_idx)
    print(f'indicies should be in closed range: [{anomaly_start}, {anomaly_stop}]')
    print('done')
    # ----- /anomaly detection ------

    plot(original, n_train, y_test, y_hat, title="Detection Result",
         start=anomaly_start, stop=anomaly_stop, anomalies_found=anomalies_idx, input_size=input_size)
    # s.plot_losses()


plt.show()


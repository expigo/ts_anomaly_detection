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

import matplotlib
matplotlib.use("TkAgg")


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


d = data_utils.Dataset.HEXAGON
hidx = 22
data_utils.plot_hexagon_location_by_id(hidx)
# plt.show()

scaler = model_utils.get_scaler('minmax')

# 1. get data
original, scaled, n_train, anomaly_start, anomaly_stop, name = data_utils.get_data(d, scaler=scaler, hidx=hidx)


def train_evaluate(hps):
    print(hps)

    input_size = int(hps[0])  # window size
    output_size = 1

    N_EPOCHS = int(hps[1])

    lr = hps[2]
    batch_size = int(hps[3])
    hidden_dim = int(hps[4])
    n_layers = int(hps[5])
    # af = 'tanh'
    dropout = 0
    weight_decay = 1e-6


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

    # 2. get dataloaders
    train_loader, train_loader_one, test_loader, test_loader_one, val_loader, train_X \
        = model_utils.get_dataloaders(scaled, batch_size, input_size, output_size, n_train=n_train, val_set=val_set)

    # 3. get_model
    model = model_utils.get_model('gru', model_params)

    # # 4. train
    # trained, history = train(rnn, N_EPOCHS, train_loader, val_loader=val_loader)
    # model_utils.save_model(trained, history, model_params, training_params, d, hidx, val_set=val_set)
    # # or use trained one
    # trained = model_utils.load_model()
    #
    # #
    # # # 5. evaluate
    # y_test, y_hat = evaluate(trained, test_loader_one, scaler)
    # y_train, y_hat_train = evaluate(trained, train_loader_one, scaler)
    # plot(original, 0, y_train, y_hat_train, title="training")

    # 4s.
    loss_fn = torch.nn.MSELoss(reduction="sum")
    # loss_fn = torch.nn.MSELoss(reduction="mean")
    # loss_fn = torch.nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    s = Sensei(model, optimizer, loss_fn)

    # 5s.
    trained, history = s.train(N_EPOCHS, train_loader, val_loader)
    # s.plot_losses()

    # s, trained = Sensei.from_trained('model_20211022-152944')
    #
    y_test, y_hat = s.evaluate(
        test_loader=train_loader_one
    )

    y_test = scaler.inverse_transform(y_test)
    y_hat = scaler.inverse_transform(y_hat)

    mae = mean_absolute_error(y_test, y_hat)
    rmse = mean_squared_error(y_test, y_hat) ** .5
    r2 = r2_score(y_test, y_hat)
    # print(mae, rmse, r2)

    return mae


from hpo.pso import PSO

# af = 'tanh'
dropout = 0
weight_decay = 1e-6

N = 10
SS = 8
pso = PSO(swarm_size=SS, N=N)
position, fitness = pso.run(fitness_fn=lambda X: train_evaluate(X),
                                space=[{
                                "low": 50,  # input_dim
                                "high": 100,
                                "type": "discrete",
                                "repeat": 1
                            },
                            {
                                "low": 100,  # number of epochs
                                "high": 200,
                                "type": "discrete",
                                "repeat": 1
                            },
                                {
                                    "low": 1e-8,  # lr
                                    "high": .1,
                                    "type": "continuous",
                                    "repeat": 1
                                },
                                {
                                    "low": 4,  # batch size
                                    "high": 128,
                                    "type": "discrete",
                                    "repeat": 1
                                },
                                {
                                    "low": 1,  # hidden_dim
                                    "high": 32,
                                    "type": "discrete",
                                    "repeat": 1
                                },
                                {
                                    "low": 1,  # n_layers
                                    "high": 32,
                                    "type": "discrete",
                                    "repeat": 1
                                },
                            ])

print(pso.best_position_history[-1].raw, pso.best_fitness_history[-1])
pso.plot_fitness_history()
plt.show()

# plot(original, n_train, y_test, y_hat, title="test")
# # ----- anomaly detection ------
# def reconstruct_ts(model, dataloder):
#
#     # trained = model.eval()
#     # y_train_one, _ = trained(train_loader_one.dataset.tensors[0], None)
#     # train_mae_one = np.abs(train_loader_one.dataset.tensors[1].detach().numpy().ravel() -
#     #                        y_train_one.detach().numpy().ravel())
#
#     predictions = []
#     # criterion = torch.nn.L1Loss(reduction='sum')
#     criterion = torch.nn.MSELoss(reduction='sum')
#     losses = []
#     for x, y in dataloder:
#         model = model.eval()
#         yhat, _ = model(x, None)
#
#         losses.append(criterion(y, yhat).item())
#         predictions.append(yhat.detach().numpy().ravel())
#
#     return np.array(predictions), np.array(losses)
#
#
# def get_threshold(losses: np.array, kde=False):
#     bins = np.linspace(losses.min(), losses.max(), 50)
#     bin_nums = np.digitize(losses, bins) - 1
#     hist_vals = np.bincount(bin_nums)
#
#     gt = losses
#     if kde is not None:
#
#         if isinstance(kde, float):
#             kde = KernelDensity(bandwidth=.1).fit(losses[:, None])
#         else:
#             params = {'bandwidth': np.logspace(-1, 1, 100)}
#             grid = GridSearchCV(KernelDensity(), params)
#             grid.fit(losses[:, None])
#
#             kde = grid.best_estimator_
#
#         # losses = losses.reshape(-1, 1)
#         scores = kde.score_samples(losses[:, None])
#         sns.displot(scores, bins=50, kde=True)
#         threshold = np.quantile(scores, .0000001)
#         gt = scores
#     else:
#         kde = None
#         threshold = losses.max()
#         print('max mae:', threshold, 'at', np.argwhere(losses == losses.max()))
#
#
#     return threshold, kde, gt
# def detect_anomalies(model, train_data, test_data, kde=True):
#     reconstructed_train_data, losses = reconstruct_ts(model, train_data)
#     reconstructed_test_data, losses_test = reconstruct_ts(model, test_data)
#
#     THRESHOLD, kde_test, train_scores = get_threshold(losses, kde=kde)
#
#     if kde:
#         losses_test = kde_test.score_samples(losses_test.reshape(-1, 1))
#
#     anomalies_count = sum(l <= THRESHOLD for l in losses_test)
#     print(f'Number of anomalies found: {anomalies_count}')
#     anomalies_idxs = np.argwhere(losses_test <= THRESHOLD) + n_train
#     return anomalies_idxs
#
#
# anomalies_idx = detect_anomalies(trained, train_loader_one, test_loader_one) #, kde=.1)
# print(anomalies_idx)
# print(f'indicies should be in closed range: [{anomaly_start}, {anomaly_stop}]')
# print('done')
# # ----- /anomaly detection ------
#
# plt.show()
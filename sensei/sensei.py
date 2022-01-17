import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KernelDensity
import seaborn as sns

import copy
from collections import namedtuple
from datetime import datetime

from hpo.hyperparams import Hyperparams, HP_TYPE
import models.utils as u
import data_utils.utils as d

u.set_seed(42)
plt.style.use('bmh')


class Sensei:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history_per_epoch = dict(train=[], val=[])
        self.history_per_batch = dict(train=[], val=[])

    @staticmethod
    def from_trained(model, loss_fn):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.hps['lr'])  #, weight_decay=hps['weight_decay'])
        loss_fn = u.get_loss_fn(loss_fn)

        return Sensei(model, None, loss_fn)

    def save_model(self, dataset_desc, model_info, hps, val_set: bool, root='',
                   name:str = 'model_' + datetime.now().strftime("%Y%m%d-%H%M%S")):
        print('Saving the model...')
        name = u.save_model_with_hps(model=self.model,
                                     history=self.history_per_epoch,
                                     model_info=model_info,
                                     model_params=hps.get_model_params(),
                                     training_params=hps.get_params_by_type(kind=HP_TYPE.TRAINING),
                                     dataset_desc=dataset_desc,
                                     val_set=val_set,
                                     root=root,
                                     name=name)
        print(f'The model has been saved! {name}')

    @staticmethod
    def from_hps(self, model_name, hps: Hyperparams, train_loader, val_loader=None):
        model = u.get_model(model_name, hps.get_model_params())

        # loss_fn = torch.nn.MSELoss(reduction="sum")
        # loss_fn = torch.nn.MSELoss(reduction="mean")
        # loss_fn = torch.nn.L1Loss(reduction="sum")
        # loss_fn = torch.nn.L1Loss(reduction="mean")
        # loss_fn = u.RMSELoss()
        loss_fn = u.get_loss_fn('rmse')
        optimizer = torch.optim.Adam(model.parameters(), lr=hps['lr'])  # , weight_decay=hps['weight_decay'])

        sensei = Sensei(model, optimizer, loss_fn)
        sensei.train(hps["n_epochs"], train_loader, val_loader)

        return sensei

    def train(self, n_epochs, train_loader, val_loader=None):

        hidden = None  # initial hidden

        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf

        n_epochs = int(n_epochs)
        for epoch in range(1, n_epochs + 1):
            self.model = self.model.train()

            batch_losses = []
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output, hidden = self.model(data, hidden)
                hidden = hidden.data
                loss = self.loss_fn(output, target)  # squeeze?
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                self.history_per_batch["train"].append(loss.item())

            # valid
            val_loss = []
            if val_loader is not None:
                self.model = self.model.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        output, hidden = self.model(data, None)
                        hidden = hidden.data
                        loss = self.loss_fn(output, target)
                        val_loss.append(loss.item())
                        self.history_per_batch["val"].append(loss.item())


            train_loss = np.mean(batch_losses)
            history['train'].append(train_loss)

            # message info printing
            if val_loader is not None:
                val_loss = np.mean(val_loss)
                history['val'].append(np.mean(val_loss))
                val_msg = f" |  Avg val loss {np.mean(val_loss)}"
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            else:
                val_msg = ""
                history['val'].append(np.nan)

            msg = f"Epoch: [{epoch}/{n_epochs}]  |" \
                  f" Average training Loss: {np.mean(train_loss)} {val_msg}"
            print(msg)

        if val_loader is not None:
            self.model.load_state_dict(best_model_wts)

        self.history_per_epoch = history
        return self.model.eval(), history

    def evaluate(self, test_loader, scaler=None, residuals=False, plot=False, title="Result"):
        predictions = []
        values = []
        diffs = []

        if residuals:
            with torch.no_grad():
                self.model = self.model.eval()
                for x, y in test_loader:
                    y_hat, _ = self.model(x, None)
                    diffs.append(self.loss_fn(y, y_hat).item())
                    # TODO: the code below most probably works fine only for out_dim=1: closer investigation required
                    y_hat = y_hat.detach().numpy().reshape(-1, 1)
                    y = y.detach().numpy().reshape(-1, 1)

                    predictions.append(y_hat)
                    values.append(y)
        else:
            with torch.no_grad():
                self.model = self.model.eval()
                for x, y in test_loader:
                    y_hat, _ = self.model(x, None)
                    # TODO: the code below most probably works fine only for out_dim=1: closer investigation required
                    y_hat = y_hat.detach().numpy().reshape(-1, 1)
                    y = y.detach().numpy().reshape(-1, 1)

                    predictions.append(y_hat)
                    values.append(y)

        y_test = np.array(values).reshape(-1, 1)
        y_hat = np.array(predictions).reshape(-1, 1)

        if scaler is not None:
            y_test = scaler.inverse_transform(y_test)
            y_hat = scaler.inverse_transform(y_hat)

        if plot:
            fig, ax = plt.subplots()

            ax.set_title(f'{title}')
            ax.plot(y_test, label='y_test')
            ax.plot(y_hat, label='y_hat')

        if residuals:
            return y_test, y_hat, np.array(diffs), calc_stats(y_test, y_hat)
        else:
            return y_test, y_hat, calc_stats(y_test, y_hat)

    def plot_losses(self):
        plt.plot(self.history_per_batch['train'], label="Training loss")
        plt.plot(self.history_per_batch['val'], label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def detect_anomalies(self, ts, train_loader_one, test_loader_one, input_size, kde=True,
                         include_n_highest_residuals:int=2):
        print('Anomaly Detection Started!')
        y_train, reconstructed_train_data, residuals, metrics_train = self.evaluate(train_loader_one, residuals=True,
                                                                                 scaler=ts.scaler)
        y_test, reconstructed_test_data, residuals_test, metrics_test = self.evaluate(test_loader_one, residuals=True,
                                                                                   scaler=ts.scaler)

        # qs = [.1, .01, .0005]
        # qs = [.1, .01, .00025]
        qs = [.01]
        THRESHOLDs, kde_test, train_scores = _get_threshold(residuals, include_last_n=include_n_highest_residuals, kde=kde, q=qs)

        if kde:
            residuals_test = kde_test.score_samples(residuals_test.reshape(-1, 1))
            # plt.close(); plt.plot(residuals_test, np.zeros_like(residuals_test) + 0., marker='x', ms=2); plt.plot(residuals_test[anomalies_idxs.ravel() - n_train], np.zeros_like(residuals_test[anomalies_idxs.ravel() - n_train]) + 0., 'x', color='black', ms=20); plt.show()

        _plot_scores_and_thresholds(THRESHOLDs, include_n_highest_residuals, qs, residuals_test, ts)

        anomalies = []

        # quantiles
        for i, q in enumerate(qs):
            anomalies.append(
                _hunt_anomalies(THRESHOLDs[i], input_size, reconstructed_test_data, residuals_test, ts, y_test,
                                title=f"Detection Result [quantile: {q}]]")
            )

        # nth residuals
        for j in range(1, include_n_highest_residuals + 1):
            anomalies.append(
                _hunt_anomalies(THRESHOLDs[j], input_size, reconstructed_test_data, residuals_test, ts, y_test,
                                title=f"Detection Result [ith kde score as threshold: i={j}]")
            )


        plt.show()
        qs.extend(THRESHOLDs[-include_n_highest_residuals:])
        return dict(zip(qs, anomalies))


    def detect_anomalies_basic(self, ts, train_loader_one, test_loader_one, input_size,
                               include_n_highest_residuals:int=2, plot=True):
        print('Anomaly Detection Started!')
        y_train, reconstructed_train_data, residuals, metrics_train = self.evaluate(train_loader_one, residuals=True,
                                                                                 scaler=ts.scaler)
        y_test, reconstructed_test_data, residuals_test, metrics_test = self.evaluate(test_loader_one, residuals=True,
                                                                                   scaler=ts.scaler)

        # qs = [.1, .01, .0005]
        # qs = [.1, .01, .00025]
        qs = [.01]  # sorting is handled by numpy

        THRESHOLDs = _get_threshold_basic(residuals=residuals, include_last_n=include_n_highest_residuals, qs=qs)

        if plot:
            _plot_scores_and_thresholds(THRESHOLDs, include_n_highest_residuals, qs, residuals_test, ts)

        anomalies = []

        # quantiles
        for i, q in enumerate(qs):
            anomalies.append(
                _hunt_anomalies(THRESHOLDs[i], input_size, reconstructed_test_data, residuals_test, ts, y_test,
                                title=f"Detection Result [quantile: {q}]]", plot=plot)
            )

        # nth residuals
        for j in range(1, include_n_highest_residuals + 1):
            anomalies.append(
                _hunt_anomalies(THRESHOLDs[j], input_size, reconstructed_test_data, residuals_test, ts, y_test,
                                title=f"Detection Result [ith kde score as threshold: i={j}]", plot=plot)
            )


        plt.show()
        qs.extend(THRESHOLDs[-include_n_highest_residuals:])
        return dict(zip(qs, anomalies))



def _hunt_anomalies(threshold,  input_size, reconstructed_test_data,
                    residuals_test, ts, y_test, title="Detection Result", plot=True):
    anomalies_count = sum(l >= threshold for l in residuals_test)
    # l = residuals_test[residuals_test <= threshold]
    anomalies_idxs = np.argwhere(residuals_test >= threshold) + ts.n_train + input_size
    # anomalies_with_scores = list(zip(l, anomalies_idxs.ravel()))
    if plot:
        _plot_ad_results(ts, y_test, reconstructed_test_data,
             title=title,
             anomalies_found=anomalies_idxs,
             input_size=input_size)
        # plt.close()
        # plt.show()

    # anomaly detection -> antek
    confusion_matrix, comp_score = _calc_ad_metrics(anomalies_idxs, ts, y_test)
    print(f">>>> threshold: {threshold}")
    print(f'Number of anomalies found: {anomalies_count}')
    print(f"confusion matrix: {confusion_matrix}")
    print(f"competition score: {(comp_score * 100):.2f}%")
    print("<<<<<")


    return anomalies_idxs, confusion_matrix, comp_score


def _calc_ad_metrics(anomalies_idxs, ts, y_test):
    X = set(np.arange(ts.n_train, ts.n_train + len(y_test)))
    GT = set(np.arange(ts.anomaly_start, ts.anomaly_stop))
    GT_complement = X.difference(GT)
    P = set(anomalies_idxs.ravel())
    P_complement = X.difference(P)
    TP = set.intersection(P, GT)
    FP = set.intersection(P, GT_complement)
    FN = set.intersection(P_complement, GT)
    TN = set.intersection(P_complement, GT_complement)

    confusion_matrix = [[len(TP), len(FP)], [len(FN), len(TN)]]
    denom = len(TP) + len(FP)
    if denom != 0:
        competition_score = len(TP) / denom
    else:
        competition_score = 0

    return confusion_matrix, competition_score


def _plot_scores_and_thresholds(THRESHOLDs, n_const, qs, residuals_test, ts):
    fig, ax = plt.subplots()
    ax.plot(residuals_test, np.zeros_like(residuals_test) + 0., marker='o', ms=10, linestyle="None",
                        label='all scores')
    ax.plot(residuals_test[np.arange(ts.anomaly_start - ts.n_train, ts.anomaly_stop - ts.n_train)],
                        np.zeros_like(
                residuals_test[np.arange(ts.anomaly_start - ts.n_train, ts.anomaly_stop - ts.n_train)]) + 0.,
            'H', ms=6, label='GT')

    dashlist = [(5, 2), (2, 5), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2)]
    from matplotlib.cm import get_cmap
    name = "tab10"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    ax.set_prop_cycle(color=colors)
    for i in range(len(qs)):
        ax.axvline(THRESHOLDs[i],
                   ls='--',
                   # dashes=dashlist[i],
                   # c='black',
                   c=cmap(i),
                   ymin=0.4, ymax=0.6,
                   label=f'threshold: q={qs[i]} value: {THRESHOLDs[i]:.4f}')

    for j in range(len(qs), n_const + len(qs)):
        ax.axvline(THRESHOLDs[-j],
                   ls='-',
                   # dashes=dashlist[i],
                   c='black',
                   ymin=0.3, ymax=0.7,
                   label=f'ith highest residual: [i={j} value: {THRESHOLDs[-j]:.4f}]')


    ax.set_xlabel('log-likelihood of each sample under the model')
    ax.yaxis.set_visible(False)
    ax.grid(None)
    ax.set_ylim([-0.1, 0.1])
    plt.title(f'Scores and Foundings')
    plt.legend()


def calc_stats(gt, y_hat, scaler=None):
    if scaler is not None:
        gt = scaler.inverse_tarnsform(gt)
        y_hat = scaler.inverse_tarnsform(y_hat)

    mae = mean_absolute_error(gt, y_hat)
    mse = mean_squared_error(gt, y_hat)
    rmse = mean_squared_error(gt, y_hat) ** .5
    r2 = r2_score(gt, y_hat)

    metrics = namedtuple('metrics', ['mae', 'mse', 'rmse', 'r2'])

    return metrics(mae, mse, rmse, r2)


def _plot_ad_results(ts, y_test, y_hat, anomalies_found, input_size, title="result"):
    input_size = int(input_size)
    anomalies_found = anomalies_found.ravel().astype(int)
    X = np.arange(ts.n_train + input_size, len(y_test) + ts.n_train + input_size)
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.scatter(x=X, y=y_test, label='y')
    # ax.scatter(x=X, y=y_hat, label='y_hat')

    ax.axvspan(ts.anomaly_start, ts.anomaly_stop, alpha=0.2, color='salmon', label='expected anomaly occurrence area')
    # ax.scatter(x=anomalies_found, y=dataset[anomalies_found+input_size],
    #            marker='x', c='purple', label='anomalies found - input_sizex')
    # ax.plot(dataset[input_size:], label="dataset")
    # ax.plot(dataset, label="dataset")
    ax.plot(X, y_test, label='ground truth')
    ax.plot(X, y_hat, label='predictions')
    # print(anomalies_found, ts.n_train, input_size)
    ax.scatter(x=anomalies_found, y=y_test[anomalies_found - ts.n_train - input_size],
               marker='x', c='black', label='anomalies detected')
    # ax.scatter(x=anomalies_found+input_size, y=dataset[anomalies_found],
    #            marker='x', c='black', label='anomalies detected')
    # ax.scatter(x=anomalies_found+input_size, y=y_test[anomalies_found-test_idx+input_size],
    #            marker='o', c='yellow', label='anomalies detected shifted')
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    # plt.show()


def _get_threshold_basic(residuals: np.array, include_last_n:int=2, qs=None, plot=True):

    if plot:
        # sns.displot(data=residuals, bins=50, kde=True)
        sns.displot(data=residuals, kde=True)
        # sns.displot(data=residuals, kind='kde')
    t = []

    residuals = np.sort(residuals)[::-1]

    if qs is not None:
        for q in qs:
            t.append(np.quantile(residuals, 1-q))

    t.extend(residuals[:include_last_n])
    return t


def _get_threshold(residuals: np.array, kde=False, include_last_n:int=2, q=[0.01]):
    # bins = np.linspace(losses.min(), losses.max(), 50)
    # bin_nums = np.digitize(losses, bins) - 1
    # hist_vals = np.bincount(bin_nums)

    gt = residuals
    if kde is not None:

        if isinstance(kde, float):
            kde = KernelDensity(bandwidth=.1).fit(residuals[:, None])
        else:
            params = {'bandwidth': np.logspace(-1, 1, 100)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(residuals[:, None])

            kde = grid.best_estimator_

        # losses = losses.reshape(-1, 1)
        scores = kde.score_samples(residuals[:, None])
        sns.displot(scores, bins=50, kde=True)
        thresholds = np.quantile(scores, q)
        thresholds = np.append(thresholds, np.sort(scores)[:include_last_n])
        gt = scores
    else:
        kde = None
        # thresholds = losses.max()
        thresholds = residuals[-include_last_n:]  # TODO: sort first!
        sns.displot(residuals, bins=50, kde=True)

    return thresholds, kde, gt


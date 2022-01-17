import torch
from matplotlib import pyplot as plt

import models.utils as mutils
from sensei.sensei import Sensei
import data_utils.utils as dutils
from data_utils.dataset_wrapper import TimeSeriesDataset

hidx = 4

# pso_name = 'train/pso_16_10_20220115-022127_gru_mae_tanh'
# pso_name = 'pso_16_10_20220114-030724_gru_rmse_tanh'
# pso_name = 'pso_16_10_20220114-150227_lstm_mse_tanh'
# pso_name = 'pso_16_10_20220114-164608_rnn_mse_tanh'
# pso_name = 'val/pso_32_16_20220117-121814_gru_mae_tanh'
pso_name = 'val/pso_32_16_20220117-134940_gru_mae_tanh'

model_name, loss_fn, af = pso_name.split('_')[-3:]

# loss_fn = mutils.get_loss_fn(loss_fn)

models = mutils.get_all_models_from_dir(dir_name=pso_name)

senseis = []
metrics = []
ad = []

plot = False

for i, (model, model_params, training_params, loss_history) in enumerate(models):
    print(f'--------- model no. {i} ----------')
    sensei = Sensei.from_trained(model, loss_fn)
    ts, dataset_description = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, hidx, 'minmax')

    dataloaders = ts.get_dataloaders(batch_size=training_params['batch_size'], input_size=model_params['input_size'])

    results_train = sensei.evaluate(test_loader=dataloaders.train_loader_one,
                                    plot=plot, title="Train Set Regression", scaler=ts.scaler)
    metrics_train_set = results_train[-1]

    results_val = sensei.evaluate(test_loader=dataloaders.val_loader_one,
                                  plot=plot, title='Validation Set Regression', scaler=ts.scaler)
    metrics_val_set = results_val[-1]

    results_test = sensei.evaluate(test_loader=dataloaders.test_loader_one,
                                   plot=plot, title='Test Set Regression', scaler=ts.scaler)
    metrics_test_set = results_test[-1]

    metrics.append((metrics_train_set, metrics_val_set, metrics_test_set))

    print(f'>>> train rmse: {metrics_train_set.rmse}')
    print(f'>>> val rmse: {metrics_val_set.rmse}')
    print(f'>>> test rmse: {metrics_test_set.rmse}')

    fig, ax = plt.subplots()
    ax.plot(loss_history['train'], label="mean train loss per epoch")
    ax.plot(loss_history['val'], label="mean val loss per epoch")
    plt.legend()
    # plt.show()

    results_ad = sensei.detect_anomalies_basic(ts,
                                               dataloaders.train_loader_one,
                                               dataloaders.test_loader_one,
                                               input_size=model_params['input_size'],
                                               plot=plot)

    ad.append(results_ad)
    print(f'--------- /{i} ----------')


from itertools import islice
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

# get best comp score, train set & val set losses
comp_score_per_model = []
train_evals = []
val_evals = []

for i, res in enumerate(ad):
    comp_score_per_model.append(res[next(islice(iter(res), 1, 1+1))][2] * 100)
    # rmse
    train_evals.append(metrics[i][0].rmse)
    val_evals.append(metrics[i][1].rmse)

df = pd.DataFrame(
    {'train_loss': train_evals,
     'val_loss': val_evals,
     'ad_score': comp_score_per_model
     })

import matplotlib
from  matplotlib.colors import LinearSegmentedColormap

c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
v = [0,.15,.4,.5,0.6,.9,1.]
l = list(zip(v,c))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
# fig,ax = plt.subplots(1)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.scatter(df.val_loss, df.train_loss, s=100, c=df.ad_score, cmap=cmap)
plt.xlabel('Validation Set Regression [RMSE]', fontsize=18)
plt.ylabel('Training Set Regression [RMSE]', fontsize=16)
norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

plt.show()
print()
#%%

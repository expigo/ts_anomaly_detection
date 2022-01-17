import torch

import models.utils as mutils
from sensei.sensei import Sensei
import data_utils.utils as dutils
from data_utils.dataset_wrapper import TimeSeriesDataset

hidx = 4

# pso_name = 'train/pso_16_10_20220115-154642_gru_mae_tanh'
# pso_name = 'pso_16_10_20220114-030724_gru_rmse_tanh'
# pso_name = 'pso_16_10_20220114-150227_lstm_mse_tanh'
pso_name = 'val/pso_32_16_20220117-134940_gru_mae_tanh'

model_name, loss_fn, af = pso_name.split('_')[-3:]

# loss_fn = mutils.get_loss_fn(loss_fn)

models = mutils.get_all_models_from_dir(dir_name=pso_name)

model, model_params, training_params, loss_history = models[-5]

sensei = Sensei.from_trained(model, loss_fn)
ts, dataset_description = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, hidx, 'minmax')

dataloaders = ts.get_dataloaders(batch_size=training_params['batch_size'], input_size=model_params['input_size'])

sensei.detect_anomalies_basic(ts,
                              dataloaders.train_loader_one,
                              dataloaders.test_loader_one,
                              input_size=model_params['input_size'],
                              include_n_highest_residuals=5, plot=True)

import matplotlib.pyplot as plt
plt.show()

print()

#%%

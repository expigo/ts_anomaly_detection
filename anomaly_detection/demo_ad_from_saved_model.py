import torch

import models.utils as mutils
from sensei.sensei import Sensei
import data_utils.utils as dutils
from data_utils.dataset_wrapper import TimeSeriesDataset

# hidx = 5

# pso_name = 'train/pso_16_10_20220115-154642_gru_mae_tanh'
# pso_name = 'pso_16_10_20220114-030724_gru_rmse_tanh'
# pso_name = 'pso_16_10_20220114-150227_lstm_mse_tanh'
# pso_name = 'val/pso_32_16_20220117-134940_gru_mae_tanh'
# pso_name = 'train/pso_4_4_20220117-183324_gru_mae_tanh'
# pso_name = 'val/final_hopefully/pso_16_10_20220118-015555_gru_mae_tanh'
# pso_name = '4-report_draft/val/pso_16_10_20220118-015555_gru_mae_tanh'
# pso_name = '4-report_draft/train/pso_16_10_20220118-033403_gru_mae_tanh'

# pso_name = '27-report_draft/train/pso_16_10_20220118-135738_gru_mae_tanh'
# pso_name = '27-report_draft/val/pso_16_10_20220118-120915_gru_mae_tanh'

# pso_name = 'ok/4-report_draft/val/pso_16_10_20220118-120915_gru_mae_tanh'

# root = './4/not_shuffled/train'
# pso_name = f'pso_16_10_20220120-135348_gru_mae_tanh'

# root = './27/not_shuffled/val'
# pso_name = f'pso_16_10_20220120-184158_lstm_mae_tanh'

# root='./4/not_shuffled/train'
# pso_name = 'pso_16_10_20220120-135348_gru_mae_tanh'

# root='./4/not_shuffled/train'
# pso_name = 'pso_16_10_20220120-144739_lstm_mae_tanh'

# root = './27/not_shuffled/val'
# pso_name = 'pso_16_10_20220120-183012_gru_mae_tanh'

# root = './27/not_shuffled/val'
# pso_name = 'pso_16_10_20220120-183047_gru_mse_tanh'

# root = './27/not_shuffled/val'
# pso_name = 'pso_16_10_20220120-184158_lstm_mae_tanh'

# root = './90/not_shuffled/train'
# pso_name = 'pso_16_10_20220121-005151_gru_mse_tanh'

# hidx = 90
# root = './90/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-045114_gru_mae_tanh'

# hidx = 5
# root = './5/not_shuffled/train'
# pso_name = 'pso_16_10_20220121-133127_gru_mse_tanh'
#
# hidx = 5
# root = './5/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-211343_gru_mse_tanh'

# hidx = 5
# root = './5/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-211343_gru_mse_tanh'


# hidx = 27
# root = './27/not_shuffled/train'
# pso_name = 'pso_16_10_20220121-210349_rnn_mse_relu'

# hidx = 90
# root = './90/not_shuffled/val'
# pso_name = 'pso_16_10_20220121-045114_gru_mae_tanh'

# hidx = 90
# root = './90/not_shuffled/train'
# pso_name = 'pso_16_10_20220122-044126_lstm_mse_tanh'

# hidx = 151
# root = './151/not_shuffled/val'
# pso_name = 'pso_16_10_20220122-235729_gru_mse_tanh'

# hidx = 151
# root = './151/not_shuffled/train'
# pso_name = 'pso_16_10_20220122-235751_gru_mse_tanh'
#
# hidx = 5
# root = 'yet_another_careless_mistake_fixed/5/train'
# pso_name = 'pso_16_10_20220124-002719_gru_mse_tanh'

# hidx = 29
# root = 'yet_another_careless_mistake_fixed/29/val'
# pso_name = 'pso_16_10_20220124-161650_gru_mae_tanh'

# hidx = 34
# root = 'yet_another_careless_mistake_fixed/34/val'
# pso_name = 'pso_16_10_20220124-161258_gru_mae_tanh'

# hidx = 127  #  [4] works for 34
# root = 'yet_another_careless_mistake_fixed/127/val'
# pso_name = 'pso_16_10_20220125-192605_gru_mae_tanh'

# hidx = 4
# root = 'yet_another_careless_mistake_fixed/4/train'
# pso_name = 'pso_16_10_20220120-135348_gru_mae_tanh'

# hidx = 77
# root = 'yet_another_careless_mistake_fixed/77/val'
# pso_name = 'pso_16_10_20220126-053441_gru_mae_tanh'

# hidx = 90
# root = 'yet_another_careless_mistake_fixed/90/train'
# pso_name = 'pso_16_10_20220127-131848_gru_mae_tanh'
#
# hidx = 4
# root = 'yet_another_careless_mistake_fixed_again/4/train'
# pso_name = 'pso_16_10_20220128-234849_rnn_mse_tanh'
#
# hidx = 34
# root = 'yet_another_careless_mistake_fixed_again/34/train'
# pso_name = 'pso_16_10_20220129-032752_rnn_mse_tanh'

hidx = 248
root = 'yet_another_careless_mistake_fixed_again/248/val'
pso_name = 'pso_16_10_20220130-235947_rnn_mae_relu'

model_name, loss_fn, af = pso_name.split('_')[-3:]

# loss_fn = mutils.get_loss_fn(loss_fn)

models, model_dir_names = mutils.get_all_models_from_dir(dir_name=f'{root}/{pso_name}')

model, model_params, training_params, loss_history = models[-1]

sensei = Sensei.from_trained(model, loss_fn)
ts, dataset_description = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, hidx, 'minmax')

dataloaders = ts.get_dataloaders(batch_size=training_params['batch_size'], input_size=model_params['input_size'])

import matplotlib
matplotlib.use('TkAgg')

# sensei.detect_anomalies_test_set_based_threshold(ts,
#                                                  # dataloaders.train_loader_one,
#                                                  dataloaders.test_loader_one,
#                                                  input_size=model_params['input_size'],
#                                                  include_n_highest_residuals=2, plot=True)

sensei.detect_anomalies_train_set_based_threshold(ts,
                                                 dataloaders.train_loader_one,
                                                 dataloaders.test_loader_one,
                                                 input_size=model_params['input_size'],
                                                 include_n_highest_residuals=2, plot=True)

import matplotlib.pyplot as plt
plt.show()

print()


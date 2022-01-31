from matplotlib import pyplot as plt
import torch

from data_utils.dataset_wrapper import TimeSeriesDataset
from hpo.hyperparams import Hyperparams, HP_TYPE
from sensei.sensei import Sensei
import models.utils as mutils
import data_utils.utils as dutils

# 1. get data
hidx = 4
ts, dataset_description = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, hidx, 'minmax')

# 2. prepare hyperparameters
hps = Hyperparams() \
    .add_discrete('input_size', 7, 100) \
    .add_discrete('hidden_dim', 1, 32) \
    .add_discrete('n_layers', 1, 32) \
    .add_constant('output_size', 1) \
    .add_discrete('n_epochs', 3, 5, HP_TYPE.TRAINING) \
    .add_discrete('batch_size', 4, 128, HP_TYPE.TRAINING) \
    .add_cont('lr', 1e-08, 0.1, HP_TYPE.TRAINING)


# hp = hps.match([38, 22, 1, 1, 407, 92, 0.018817])
hp = hps.match([76, 29, 1, 1, 4, 29, 0.06253199039756516])


model_info = {
    'type': 'gru',
    'activation_function': 'tanh', # # TODO: it's the only option for lstm & gru models - saving should be automated
    'fitness_fn': 'rmse'
}


val_set = True
dataloaders = ts.get_dataloaders(batch_size=hp['batch_size'], input_size=hp['input_size'], val_set=val_set)

sensei = Sensei.from_hps(model_name='gru', hps=hp,
                         train_loader=dataloaders.train_loader, val_loader=dataloaders.val_loader)

sensei.save_model(dataset_description, model_info=model_info, hps=hp, val_set=val_set)

sensei.evaluate(test_loader=dataloaders.test_loader_one,
                         plot=True, title='Test Set Regression', scaler=ts.scaler)

sensei.evaluate(test_loader=dataloaders.train_loader_one,
                         plot=True, title="Train Set Regression", scaler=ts.scaler)

anomalies_idx = sensei.detect_anomalies_train_set_based_threshold(ts,
                                                                  dataloaders.train_loader_one,
                                                                  dataloaders.test_loader_one,
                                                                  input_size=hps['input_size'], )
# kde=False)

print(anomalies_idx)
print(f'indicies should be in closed range: [{ts.anomaly_start}, {ts.anomaly_stop}]')
print('done')
plt.show()
#%%

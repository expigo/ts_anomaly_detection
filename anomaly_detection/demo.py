from matplotlib import pyplot as plt

from data_utils.dataset_wrapper import TimeSeriesDataset
from hpo.hyperparams import Hyperparams, HP_TYPE
from hpo.pso_hpo import PSO_HPO, PSO_EVAL_MODE
from sensei.sensei import Sensei
from pso.pso import PSO
import models.utils as mutils
import data_utils.utils as dutils

# 1. get data
# shuffle = False
hidx = 12
ts, dataset_description = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, hidx, 'minmax')

# 2. prepare hyperparameters
hps = Hyperparams() \
    .add_discrete('input_size', 1, 100) \
    .add_discrete('hidden_dim', 1, 32) \
    .add_discrete('n_layers', 1, 32) \
    .add_constant('output_size', 1) \
    .add_discrete('n_epochs', 10, 50, HP_TYPE.TRAINING) \
    .add_discrete('batch_size', 4, 128, HP_TYPE.TRAINING) \
    .add_cont('lr', 1e-08, 0.1, HP_TYPE.TRAINING)

# 3. prepare pso
swarm_size = 16
N = 10

model_info = {
    'type': 'gru',
    'af': 'tanh',
    'loss_fn': 'mae'
}

# if shuffle:
#     dest_dir_name = f'{hidx}/shuffled'
# else:
#     dest_dir_name = f'{hidx}/not_shuffled'

dest_dir_name = f'yet_another_careless_mistake_fixed_again/{hidx}/'
# dest_dir_name = f'test/{hidx}/'

pso = PSO_HPO(hps, swarm_size, N, ts, ts_desc=dataset_description, dest_dir=dest_dir_name,
              eval_mode=PSO_EVAL_MODE.VAL_SET)
pso.run(model=model_info)

dataloaders = ts.get_dataloaders(
    batch_size=pso.best_hps['batch_size'],
    input_size=pso.best_hps['input_size'],
    shuffle=False
)
pso.best_senseis[-1].evaluate(
    test_loader=dataloaders.test_loader_one,
    plot=True,
    title='Test Set Regression',
    scaler=ts.scaler)

pso.best_senseis[-1].evaluate(
    test_loader=dataloaders.train_loader_one,
    plot=True,
    title="Train Set Regression",
    scaler=ts.scaler
)

results = pso.best_senseis[-1].detect_anomalies_test_set_based_threshold(
    ts,
    # dataloaders.train_loader_one,
    dataloaders.test_loader_one,
    input_size=pso.best_hps['input_size'],
    # kde=False
)

print(results)
print(f'indices should be in closed range: [{ts.anomaly_start}, {ts.anomaly_stop}]')
print('done')
plt.show()


#%%

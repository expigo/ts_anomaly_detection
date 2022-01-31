from sensei.sensei import Sensei
from data_utils.dataset_wrapper import TimeSeriesDataset
from pso.pso import PSO
from hpo.hyperparams import Hyperparams
import data_utils.utils as dutils
import models.utils as mutils

import numpy as np
import torch

from datetime import datetime
from enum import Enum
import copy


class PSO_EVAL_MODE(Enum):
    VAL_SET = 'val'
    TRAIN_SET = 'train'
    CV = 'cv'  # TODO


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


class PSO_HPO:
    def __init__(self, hps: Hyperparams, swarm_size=16, N=10,
                 ts=None, ts_desc=None, dest_dir: str = './',
                 eval_mode: PSO_EVAL_MODE = PSO_EVAL_MODE.VAL_SET):
        self.pso = PSO(swarm_size, N)
        self.hps = hps
        if not (ts and ts_desc):
            ts, ts_desc = TimeSeriesDataset.from_enum(dutils.Dataset.HEXAGON, 4, 'minmax')
        self.ts = ts
        self.ts_desc = ts_desc
        self.best_fitness = np.inf
        self.best_senseis: [Sensei] = []
        self.eval_mode = eval_mode
        self.name = f'{dest_dir}/{self.eval_mode.value}/pso_{swarm_size}_{N}_' + datetime.now().strftime("%Y%m%d-%H%M%S")




    @counted
    def _tune(self,
              X,
              model_info: dict,
              save_results=True):

        # 0. match hyperparameters

        dict_matched_values = self.hps.match(X)
        print(self.hps.get_as_dict())
        #       self.hps['batch_size'],
        #       self.hps['input_size'],
        #       self.hps['output_size'])

        # 1. get data loaders
        dls = self.ts.get_dataloaders(batch_size=self.hps['batch_size'],
                                      input_size=self.hps['input_size'],
                                      output_size=self.hps['output_size'],
                                      val_set=True)

        model_params = self.hps.get_model_params() | {key: model_info[key] for key in ['af']}
        model = mutils.get_model(model_info["type"], model_params)

        # loss_fn = torch.nn.MSELoss(reduction="sum")
        # loss_fn = torch.nn.MSELoss(reduction="mean")
        # loss_fn = torch.nn.L1Loss(reduction="sum")
        # loss_fn = torch.nn.L1Loss(reduction="mean")
        # loss_fn = mutils.RMSELoss()
        loss_fn = mutils.get_loss_fn(model_info["loss_fn"])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hps['lr'])  # , weight_decay=hps['weight_decay'])
        s = Sensei(model, optimizer, loss_fn)

        trained, history = s.train(self.hps['n_epochs'], dls.train_loader, dls.val_loader)

        if self.eval_mode is PSO_EVAL_MODE.VAL_SET:
            y_test, y_hat, losses, metrics = s.evaluate(
                test_loader=dls.val_loader_one,
                scaler=self.ts.scaler,
                residuals=True
            )
        elif self.eval_mode is PSO_EVAL_MODE.TRAIN_SET:
            y_test, y_hat, losses, metrics = s.evaluate(
                test_loader=dls.train_loader_one,
                scaler=self.ts.scaler,
                residuals=True
            )
        else:
            raise ValueError("Not implemented error")

        # current_fitness = metrics.mae
        current_fitness = getattr(metrics, model_info["loss_fn"])

        if current_fitness < self.best_fitness:
            print('The queen is dead, long live the queen!')
            self.best_fitness = current_fitness
            self.best_senseis.append(copy.deepcopy(s))
            self.best_hps = copy.deepcopy(self.hps)
            if save_results:
                self.best_senseis[-1].save_model(dataset_desc=self.ts_desc, model_info=model_info,
                                                 hps=self.best_hps, val_set=True,
                                                 root=self.name +
                                                      f'_{model_info["type"]}_{model_info["loss_fn"]}_{model_info["af"]}',
                                                 name='model_' + datetime.now().strftime(
                                                     "%Y%m%d-%H%M%S") + f'_{self._tune.calls}'
                                                 )

        return current_fitness

    def run(self, model={'type': 'gru'}, save_results=True):

        # 1. get space definition
        space_def = self.hps.build()

        # 2.
        best_hps, best_fitness = self.pso.run(fitness_fn=lambda h: self._tune(h, model, save_results), space=space_def)

        return best_hps, best_fitness

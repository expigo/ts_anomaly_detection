import torch
import numpy as np
from matplotlib import pyplot as plt

import copy

import utils as u


class Sensei:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = dict(train=[], val=[])

    def from_trained(self, name:str=None):


        u.load_model(None, name)

    def train(self, n_epochs, train_loader, val_loader=None):

        hidden = None  # initial hidden

        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf

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

            train_loss = np.mean(batch_losses)
            history['train'].append(train_loss)

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

        return self.model.eval(), history

    def evaluate(self, test_loader):
        predictions = []
        values = []

        with torch.no_grad():
            self.model = self.model.eval()
            for x, y in test_loader:
                y_hat, _ = self.model(x, None)
                # TODO: the code below most probably works fine only for out_dim=1: closer investigation required
                y_hat = y_hat.detach().numpy().reshape(-1, 1)
                y = y.detach().numpy().reshape(-1, 1)

                predictions.append(y_hat)
                values.append(y)

        return np.array(values).reshape(-1, 1), np.array(predictions).reshape(-1, 1)

    def plot_losses(self):
        plt.plot(self.history['train'], label="Training loss")
        plt.plot(self.history['val'], label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

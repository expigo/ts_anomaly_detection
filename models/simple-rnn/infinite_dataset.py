import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation

from models import utils

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, num_directions):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=(True if num_directions == 2 else False),
                          nonlinearity='tanh')
        self.out = nn.Linear(input_size, hidden_size)  # hidden_size=1 => 1 feature => 1*10 in and 1*10 out

    def forward(self, inputs, hidden=None):
        # hidden = self.__init__hidden()
        output, hidden = self.rnn(inputs, hidden)
        output = self.out(output)
        return output, hidden

    def __init__hidden(self):
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).double()
        return h_0


INPUT_SIZE = 1  # number of features (columns basically)
OUTPUT_SIZE = 1
HIDDEN_SIZE = 1  # no of features in last hidden state => no of out time-steps to predict
NUM_LAYERS = 2  # no of stacked rnn layers
SEQ_LENGTH = 50  # no of previous timestamps being take into account
N_SAMPLE_SIZE = 750
# We have 50 rows in the input. We divide the input into 5 (BATCH_SIZE) batches
# where each batch has exactly one row
BATCH_SIZE = int(N_SAMPLE_SIZE / SEQ_LENGTH)  # no of batches

N_EPOCHS = 10
N_ITERS = 100
model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE, 1).double()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = np.zeros(N_EPOCHS)

for epoch in range(N_EPOCHS):
    hidden = None

    for i in range(N_ITERS):

        _targets, _inputs = utils.get_noisy_wiggly_sine(N_SAMPLE_SIZE)

        inputs = torch.from_numpy(
            np.array(np.array_split(_inputs, BATCH_SIZE))
        ).unsqueeze(2).double()

        targets = torch.from_numpy(
            np.array(np.array_split(_targets, BATCH_SIZE))
        ).unsqueeze(2).double()

        # input sequence structure:
        # batch_first=False  =>  (seq_len, batch, input_size)
        # batch_first=True  =>  (batch,seq_len, input_size)
        outputs, hidden = model(inputs, hidden)

        # detach from the current graph to prevent from back-propagating
        # all the way through to the beginnings of time
        # (which is not possible, because the graph info is lost on variable reassignment)
        # effectively: requires_grad=False
        hidden = hidden.detach()

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss
        if i % 10 == 0:
            plt.clf()
            plt.ion()
            plt.title(f"Epoch {epoch}, iteration: {i}")
            plt.plot(torch.flatten(outputs.detach()), 'r-', linewidth=1, label='Output')
            plt.plot(torch.flatten(targets), 'c-', linewidth=1, label='Ground Truth')
            plt.plot(torch.flatten(inputs), 'g-', linewidth=1, label='Input')
            plt.legend()
            plt.draw()
            plt.pause(0.05)
            plt.show()

    if epoch > 0:
        print(epoch, loss)


def plot_learning_curve(l):
    fig, ax = plt.subplots()
    ax.plot(l)
    plt.title("Learning curve")
    plt.show()


plot_learning_curve(losses)

import torch


class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0, af='relu'):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, nonlinearity=af, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        r_out, hidden = self.rnn(x, hidden)
        # batch_size = x.size(0)
        # r_out = r_out.contiguous().view(batch_size, -1)
        output = self.fc(r_out)
        return output, hidden
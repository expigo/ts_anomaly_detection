import torch


class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout_prob=0, af='relu'):
        super(SimpleRNN, self).__init__()

        input_size = int(input_size)
        hidden_dim = int(hidden_dim)
        n_layers = int(n_layers)
        output_size = int(output_size)

        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, nonlinearity=af, batch_first=True, dropout=dropout_prob)
        self.fc = torch.nn.Linear(hidden_dim, output_size)

        # print(self.fc.weight)

    def forward(self, x, hidden):
        r_out, hidden = self.rnn(x, hidden)
        # batch_size = x.size(0)
        # r_out = r_out.contiguous().view(batch_size, -1)
        output = self.fc(r_out)
        return output, hidden
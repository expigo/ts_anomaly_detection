import torch

class GRUModel(torch.nn.Module):
    """
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_size, hidden_dim, n_layers, output_size, dropout_prob=0, af='tanh'):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_size (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            n_layers (int): The number of layers in the network
            output_size (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        input_size = int(input_size)
        hidden_dim = int(hidden_dim)
        n_layers = int(n_layers)
        output_size = int(output_size)

        self.layer_dim = n_layers
        self.hidden_dim = hidden_dim

        self.gru = torch.nn.GRU(
            input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        # out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out, h0
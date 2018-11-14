import torch
from torch import nn
import torch.nn.functional as F


class HawkesRNNModel(nn.Module):
    """
    A Hawkes model based on a simple recurrent neural network architecture.
    """

    def __init__(self, hidden_size: int):
        super(HawkesRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.RNN(1, hidden_size)
        self.decay_layer = nn.Linear(hidden_size, 1)
        self.out_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dt, hidden):
        """

        Args:
            dt: interval of time before next event
            hidden: previous hidden state

        Returns:

        """
        output, hidden = self.rnn_layer(dt, hidden)
        # Compute new decay parameter
        decay = F.sigmoid(self.decay_layer(hidden))
        output = self.out_layer(output)
        return output, hidden, decay

    def initialize_hidden(self):
        return torch.randn(self.hidden_size)

    def compute_intensity(self, hidden, s, t, decay):
        """

        Args:
            hidden:
            s: current time
            t: last event time
            decay:

        Returns:
            Intensity function value at time s.
        """
        # Compute hidden state at time s
        h_t = hidden*torch.exp(-decay*(s-t))
        return F.relu(h_t)

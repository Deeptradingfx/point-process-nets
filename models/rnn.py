import torch
from torch import nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


class HawkesRNNModel(nn.Module):
    """
    A Hawkes model based on a simple recurrent neural network architecture.

    We denote by :math:`N` the sequence lengths.
    """

    def __init__(self, hidden_size: int):
        super(HawkesRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.RNNCell(1, hidden_size, nonlinearity='relu')
        self.decay_layer = nn.Linear(hidden_size, 1)
        self.intensity_layer = nn.Linear(hidden_size, 1, bias=False)
        self.intensity_activ = nn.Softplus(beta=3)

    def forward(self, dt: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the network.

        Args:
            dt: interval of time before next event
                Shape: N * batch * 1
            hidden: previous hidden state

        Returns:

        """
        hidden = self.rnn_layer(dt, hidden)
        # Compute new decay parameter
        decay = F.softplus(self.decay_layer(hidden))
        return hidden, decay

    def initialize_hidden(self) -> Tensor:
        """

        Returns:
            Shape: batch * hidden_size
        """
        return torch.rand(1, self.hidden_size)

    def compute_intensity(self, hidden, decay, s, t) -> Tensor:
        """
        Compute the process intensity for the given parameters at the given time.

        Args:
            hidden:
            s: current time
            t: last event time
            decay: intensity decay on interval :math:`[t, \infty)`

        Returns:
            Intensity function value at time s.
        """
        # Compute hidden state at time s
        h_t = hidden*torch.exp(-decay*(s-t))
        return self.intensity_activ(self.intensity_layer(h_t))

    def compute_loss(self, times: Tensor, hiddens: Tensor, decays: Tensor, tmax: float) -> Tensor:
        """
        Negative log-likelihood.

        Number of intervals for the process, counting [0, t_1) and (t_N, tmax): N + 1

        Args:
            times: event times, including start time 0
                Shape: N + 1
            hiddens:
                Shape: (N + 1) * hidden_size
            decays:
                Shape: N + 1
            tmax: time interval bound

        Returns:

        """
        inter_times: Tensor = times[1:] - times[:-1]  # shape N
        n_times = inter_times.shape[0]
        intensity_at_event_times: Tensor = F.relu(self.intensity_layer(hiddens))
        first_term = intensity_at_event_times.log().sum(dim=0)  # scalar
        # Take uniform time samples inside of each inter-event interval
        time_samples = times[:-1] + inter_times*torch.rand_like(inter_times)  # shape N
        intensity_at_samples = torch.stack([
            self.compute_intensity(hiddens[i], decays[i], time_samples[i], times[i + 1])
            for i in range(n_times)
        ])
        integral_estimates: Tensor = inter_times*intensity_at_samples
        last_sample_time = times[-1] + (tmax - times[-1])*torch.rand(1)
        last_lambda_sample = self.compute_intensity(hiddens[-1], decays[-1], last_sample_time, times[-1])
        second_term = integral_estimates.sum(dim=0) + (tmax - times[-1])*last_lambda_sample  # scalar
        return - first_term + second_term

    def generate_sequence(self, tmax: float):
        """
        Generate an event sequence on the interval [0, tmax]

        Args:
            tmax:

        Returns:

        """
        with torch.no_grad():
            s = Tensor([0.0])
            hidden = self.initialize_hidden()
            t = s.clone()
            event_times = [t]  # record sequence start event
            event_intens = []
            decay = F.softplus(self.decay_layer(hidden))

            while s < tmax:
                max_lbda = self.get_max_lbda(hidden)
                u1 = torch.rand(1)
                # Candidate inter-arrival time
                ds = -1.0/max_lbda*u1.log()
                # update last tried time
                s = s.clone() + ds
                if s > tmax:
                    break
                u2 = torch.rand(1)
                hidden_candidate = hidden*torch.exp(-decay*(s-t))
                intens_candidate = self.intensity_activ(
                    self.intensity_layer(hidden_candidate))
                if u2.item() <= (intens_candidate/max_lbda).item():
                    # accept event
                    # update hidden state
                    hidden = hidden_candidate.clone()
                    event_intens.append(intens_candidate)
                    t = s.clone()
                    event_times.append(t)

            return event_times, event_intens

    def get_max_lbda(self, hidden: Tensor):
        """

        Args:
            hidden: previous hidden state
            t: previous event time

        Returns:

        """
        # weight of the hidden state
        wh: Tensor = self.intensity_layer.weight.data
        return F.relu(hidden.matmul(wh.abs().t()))

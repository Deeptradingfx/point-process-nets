import torch
from torch import nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


class HawkesRNNModel(nn.Module):
    """
    A Hawkes model based on a simple recurrent neural network architecture.

    Utilises a RNN Cell to update the hidden state with each incoming event.

    We denote by :math:`N` the sequence lengths.
    """

    def __init__(self, hidden_size: int):
        super(HawkesRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.Sequential(
            nn.Linear(1 + hidden_size, hidden_size),
            nn.Softplus(beta=3.0)
        )
        self.decay_layer = nn.Linear(hidden_size, 1)
        self.decay_activ = nn.Softplus(beta=3.0)
        self.intensity_layer = nn.Linear(hidden_size, 1, bias=False)
        self.intensity_activ = nn.Softplus(beta=3.0)

    def forward(self, dt: Tensor, hidden: Tensor, decay: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the network.

        Define the network parameters for the next interval :math:`[t_i,t_{i+1})`
        from the data on the current interval :math:`[t_{i-1},t_i)`.

        Args:
            dt: interval of time before event :math:`i`
                Shape: N * batch * 1
            hidden: hidden state :math:`h_i` computed after :math:`t_{i-1}` occurred
            decay: decay parameter for the current interval, computed at :math:`t_{i-1}`

        Returns:
            The hidden state and decay value for the interval, and the decayed hidden state.
            Collect them during training to use for computing the loss.
        """
        hidden_after_decay = hidden*torch.exp(-decay*dt)
        # Decay value, compute from decayed hidden state
        decay = self.decay_activ(self.decay_layer(hidden))
        # Now the event t_i occurs, we know dt has elapsed since t_{i-1}
        # Update hidden state, an event just occurred
        concat = torch.cat((dt, hidden), dim=1)
        hidden = self.rnn_layer(concat)
        return hidden, decay, hidden_after_decay

    def initialize_hidden(self) -> Tuple[Tensor, Tensor]:
        """

        Returns:
            Shape: batch * hidden_size
        """
        return torch.rand(1, self.hidden_size), torch.zeros(1)

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

    def compute_loss(self, sequence: Tensor, hiddens: Tensor, decays: Tensor, tmax: float) -> Tensor:
        """
        Negative log-likelihood.

        Number of intervals for the process, counting [0, t_1) and (t_N, tmax): N + 1

        Args:
            sequence: event sequence, including start time 0
                Shape: N + 1
            hiddens:
                Shape: (N + 1) * hidden_size
            decays:
                Shape: N + 1
            tmax: time interval bound

        Returns:

        """
        inter_times: Tensor = sequence[1:] - sequence[:-1]  # shape N
        n_times = inter_times.shape[0]
        intensity_at_event_times: Tensor = self.intensity_activ(self.intensity_layer(hiddens))
        first_term = intensity_at_event_times.log().sum(dim=0)  # scalar
        # Take uniform time samples inside of each inter-event interval
        time_samples = sequence[:-1] + inter_times * torch.rand_like(inter_times)  # shape N
        intensity_at_samples = torch.stack([
            self.compute_intensity(hiddens[i], decays[i], time_samples[i], sequence[i])
            for i in range(n_times)
        ])
        integral_estimates: Tensor = inter_times*intensity_at_samples
        last_sample_time = sequence[-1] + (tmax - sequence[-1]) * torch.rand(1)
        last_lambda_sample = self.compute_intensity(hiddens[-1], decays[-1], last_sample_time, sequence[-1])
        second_term = integral_estimates.sum(dim=0) + (tmax - sequence[-1]) * last_lambda_sample  # scalar
        return - first_term + second_term

    def generate_sequence(self, tmax: float):
        """
        Generate an event sequence on the interval [0, tmax]

        Args:
            tmax: time horizon

        Returns:
            Sequence of event times with corresponding event intensities.
        """
        with torch.no_grad():
            s = Tensor([[0.0]])
            t = s.clone()
            hidden, decay = self.initialize_hidden()
            event_times = [t]  # record sequence start event
            event_intens = []
            event_decay = []
            max_lbda = self.intensity_activ(self.intensity_layer(hidden))

            while s < tmax:
                u1 = torch.rand(1)
                # Candidate inter-arrival time
                ds = -1.0/max_lbda*u1.log()
                # update last tried time
                s = s.clone() + ds
                if s > tmax:
                    break
                u2 = torch.rand(1)
                # adaptive sampling: always update the hidden state
                hidden = hidden*torch.exp(-decay*ds)
                intens_candidate = self.intensity_activ(
                    self.intensity_layer(hidden))
                ratio = (intens_candidate/max_lbda).item()
                if u2.item() <= ratio:
                    # accept event
                    # update hidden state by running the RNN cell on it
                    hidden = self.rnn_layer(torch.cat((s-t, hidden), dim=1))
                    intens_candidate = self.intensity_activ(self.intensity_layer(hidden))
                    event_intens.append(intens_candidate)
                    # update decay
                    decay = self.decay_activ(self.decay_layer(hidden))
                    event_decay.append(decay)
                    t = s.clone()
                    event_times.append(t)
                max_lbda = intens_candidate
            event_times = torch.stack(event_times, dim=2).squeeze(0).squeeze(0)
            event_intens = torch.stack(event_intens, dim=2).squeeze(0).squeeze(0)
            event_decay = torch.stack(event_decay, dim=2).squeeze(0).squeeze(0)
            return event_times, event_intens, event_decay

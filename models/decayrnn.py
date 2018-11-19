import torch
from torch import nn
from torch import Tensor
from typing import Tuple


class HawkesDecayRNN(nn.Module):
    """
    Recurrent neural network (RNN) model using decaying hidden states between events.

    We denote by :math:`N` the sequence lengths.
    """

    def __init__(self, hidden_size: int):
        super(HawkesDecayRNN, self).__init__()
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
        from the data on the current interval :math:`[t_{i-1},t_i)`:
        :math:`\Delta t_i = t_i - t_{i-1}`, :math:`h_i` and :math:`\delta_i`.

        Args:
            dt: interval of time before event :math:`i`, :math:`\Delta t_i`
                Shape: batch * 1
            hidden: hidden state :math:`h_i` computed after :math:`t_{i-1}` occurred
                Shape: batch * hidden_size
            decay: decay parameter for the current interval, computed at the previous time :math:`t_{i-1}`
                Shape: batch * 1

        Returns:
            The hidden state and decay value for the interval, and the decayed hidden state.
            Collect them during training to use for computing the loss.
        """
        # Compute h(t_i)
        hidden_after_decay = hidden*torch.exp(-decay*dt)  # shape batch * hidden_size
        # Decay value for the next interval, predicted from the decayed hidden state
        decay = self.decay_activ(self.decay_layer(hidden_after_decay))
        # Now the event t_i occurs, we know dt has elapsed since t_{i-1}
        # Update hidden state, an event just occurred
        concat = torch.cat((dt, hidden), dim=1)  # shape batch * (input_dim + hidden_size)
        hidden = self.rnn_layer(concat)  # shape batch * hidden_size
        return hidden, decay, hidden_after_decay

    def initialize_hidden(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """

        Returns:
            Shape: batch * hidden_size, batch * 1
        """
        return torch.rand(batch_size, self.hidden_size), torch.zeros(batch_size, 1)

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
        try:
            h_t = hidden*torch.exp(-decay*(s-t))
        except Exception:
            import pdb;
            pdb.set_trace()
        return self.intensity_activ(self.intensity_layer(h_t))

    def compute_loss(self, sequence: Tensor, batch_sizes: Tensor,
                     hiddens: Tensor, decays: Tensor, tmax: float) -> Tensor:
        """
        Negative log-likelihood.

        Number of intervals for the process, counting [0, t_1) and (t_N, tmax): N + 1

        Args:
            sequence: event sequence, including start time 0
                Shape: (N + 1) * batch
            batch_sizes: batch sizes for each event sequence tensor, by length
            hiddens:
                Shape: (N + 1) * batch * hidden_size
            decays:
                Shape: N + 1
            tmax: time interval bound

        Returns:

        """
        dt_sequence: Tensor = sequence[1:] - sequence[:-1]  # shape N * batch
        n_times = len(hiddens)
        try:
            intensity_at_event_times: Tensor = [
                self.intensity_activ(self.intensity_layer(hiddens[i]))
                for i in range(n_times)
            ]  # shape N * batches
        except Exception:
            import pdb; pdb.set_trace()
        # shape batch * N
        intensity_at_event_times = nn.utils.rnn.pad_sequence(
            intensity_at_event_times, batch_first=True, padding_value=1.0)
        first_term = intensity_at_event_times.log().sum(dim=0)  # scalar
        # Take uniform time samples inside of each inter-event interval
        time_samples = sequence[:-1] + dt_sequence * torch.rand_like(dt_sequence)  # shape N
        intensity_at_samples = []
        for i in range(n_times):
            try:
                v = self.compute_intensity(hiddens[i], decays[i],
                                       time_samples[i, :batch_sizes[i]], sequence[i, :batch_sizes[i]])
            except Exception:
                print(batch_sizes[i])
            intensity_at_samples.append(v)
        intensity_at_samples = nn.utils.rnn.pad_sequence(
            intensity_at_samples, batch_first=True, padding_value=0.0)
        # import pdb; pdb.set_trace()
        integral_estimates: Tensor = dt_sequence*intensity_at_samples
        second_term = integral_estimates.sum(dim=0)
        return (- first_term + second_term).mean()

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

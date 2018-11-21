import torch
from torch import nn
from torch import Tensor
from typing import Tuple, List


class HawkesDecayRNN(nn.Module):
    """
    Recurrent neural network (RNN) model using decaying hidden states between events.

    We denote by :math:`N` the sequence lengths.


    .. math::
        h(t) = h_i e^{-\delta_i(t-t_i)}\quad t\in (t_{i-1}, t_i]
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: process dimension
            hidden_size: hidden layer dimension
        """
        super(HawkesDecayRNN, self).__init__()
        input_size += 1  # add the dimension of the beginning-of-sequence event type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, input_size)
        self.rnn_layer = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decay_layer = nn.Linear(input_size + hidden_size, 1)
        self.decay_activ = nn.Softplus(beta=4.0)
        self.intensity_layer = nn.Linear(hidden_size, input_size, bias=False)
        self.intensity_activ = nn.Softplus(beta=4.0)

    def forward(self, dt: Tensor, seq_types: Tensor, h_decay: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the network.

        Computes the network parameters for the next interval :math:`(t_i,t_{i+1}]` from:

        * the decayed hidden state :math:`h(t_i) = h_i\exp(-\delta_i(t_i-t_{i-1}))` computed
          for the interval :math:`(t_{i-1},t_i]`
        * the interval of time :math:`\Delta t_{i+1} = t_{i+1}-t_i` before the next event

        Args:
            dt: interval of time before event :math:`i`, :math:`\Delta t_i`
                Shape: batch
            seq_types: sequence event types
                Shape: batch* * input_size
            h_decay: decayed hidden state :math:`h(t_i)` at the end of :math:`(t_{i-1},t_i]`
                Shape: batch * hidden_size

        Returns:
            The hidden state and decay value for the interval, and the decayed hidden state.
            Collect them during training to use for computing the loss.
        """
        x = self.embed(seq_types)
        concat = torch.cat((x, h_decay), dim=1)
        dt = dt.unsqueeze(1)
        # Compute h(t_i)
        # Decay value for the next interval, predicted from the decayed hidden state
        # Now the event t_i occurs, we know dt has elapsed since t_{i-1}
        # Update hidden state, an event just occurred
        # concat = torch.cat((dt, hidden), dim=1)  # shape batch * (input_dim + hidden_size)
        decay = self.decay_activ(self.decay_layer(concat))
        # New hidden state h(t_i+)
        hidden = self.rnn_layer(concat)  # shape batch * hidden_size
        # decay the new hidden state to its value h(t_{i+1})
        hidden_after_decay = hidden * torch.exp(-decay * dt)  # shape batch * hidden_size
        return hidden, decay, hidden_after_decay

    def initialize_hidden(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """

        Returns:
            Shape: batch * hidden_size, batch * 1
        """
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, 1)

    def compute_intensity(self, hidden: Tensor, decay: Tensor, s: Tensor, t: Tensor) -> Tensor:
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
        # print("s: {} | t: {} | h: {}".format(s.shape, t.shape, hidden.shape))
        h_t = hidden*torch.exp(-decay*(s-t))
        return self.intensity_activ(self.intensity_layer(h_t))

    def compute_loss(self, sequence: Tensor, seq_types: Tensor, batch_sizes: Tensor,
                     hiddens: List[Tensor], decays: List[Tensor], tmax: float) -> Tensor:
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
        intensity_ev_times: Tensor = [
            self.intensity_activ(self.intensity_layer(hiddens[i]))
            for i in range(n_times)
        ]
        # shape N * batch * input_dim
        intensity_ev_times = nn.utils.rnn.pad_sequence(
            intensity_ev_times, batch_first=True, padding_value=1.0)
        ishape = intensity_ev_times.shape

        # import pdb; pdb.set_trace()
        intensity_ev_times = intensity_ev_times.reshape(
            ishape[0]*ishape[1], ishape[2]
        )[seq_types.flatten(), :].reshape(*ishape)
        first_term = intensity_ev_times.log().sum(dim=0)  # scalar
        # Take uniform time samples inside of each inter-event interval
        # sequence: Tensor = torch.cat((sequence, tmax*torch.ones_like(sequence[-1:, :])))
        # dt_sequence = sequence[1:] - sequence[:-1]
        time_samples = sequence[:-1] + dt_sequence * torch.rand_like(dt_sequence)  # shape N
        intensity_at_samples = [
            self.compute_intensity(hiddens[i], decays[i],
                                   time_samples[i, :batch_sizes[i]], sequence[i, :batch_sizes[i]])
            for i in range(n_times)
        ]
        intensity_at_samples = nn.utils.rnn.pad_sequence(
            intensity_at_samples, batch_first=True, padding_value=0.0)  # shape N * batch * input_dim
        total_intens_samples: Tensor = intensity_at_samples.sum(dim=2, keepdim=True)
        integral_estimates: Tensor = dt_sequence*total_intens_samples
        second_term = integral_estimates.sum(dim=0)
        return (- first_term + second_term).mean()

    def generate_sequence(self, tmax: float):
        """
        Generate an event sequence on the interval [0, tmax].

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
            event_hidden = []
            event_decay = []
            max_lbda: Tensor = self.intensity_activ(self.intensity_layer(hidden))

            while s < tmax:
                u1 = torch.rand(1)
                # Candidate inter-arrival time
                ds: Tensor = torch.empty_like(u1).exponential_(lambd=max_lbda.item())
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
                    t = s.clone()
                    event_times.append(t)
                    # update decay from decayed hidden state
                    decay = self.decay_activ(self.decay_layer(hidden))
                    # update hidden state by running the RNN cell on it
                    hidden = self.rnn_layer(hidden)
                    max_lbda = self.intensity_activ(self.intensity_layer(hidden))
                    event_hidden.append(hidden)
                    event_decay.append(decay)
                else:
                    max_lbda = intens_candidate
            event_times = torch.stack(event_times, dim=2).squeeze(0).squeeze(0)
            event_decay = torch.stack(event_decay, dim=2).squeeze(0).squeeze(0)
            return event_times, event_hidden, event_decay

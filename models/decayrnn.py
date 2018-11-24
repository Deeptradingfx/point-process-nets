import torch
from torch import nn
from torch import Tensor
from typing import Tuple, List
from load_synth_data import one_hot_embedding


class HawkesDecayRNN(nn.Module):
    """
    Recurrent neural network (RNN) model using decaying hidden states between events, Decay-RNN.

    We denote by :math:`N` the sequence lengths.


    .. math::
        h(t) = h_i e^{-\delta_i(t-t_i)}\quad t\in (t_{i-1}, t_i]
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
            input_size: process dimension K
            hidden_size: hidden layer dimension
        """
        super(HawkesDecayRNN, self).__init__()
        input_size += 1  # add the dimension of the beginning-of-sequence event type
        self.input_size = input_size
        self.hidden_size = hidden_size

        # self.embed = nn.Embedding(input_size, input_size)
        self.rnn_layer = nn.RNNCell(input_size, hidden_size, nonlinearity="relu")
        # self.rnn_layer = nn.Sequential(
        #     nn.Linear(input_size+hidden_size, hidden_size), nn.ReLU())
        self.decay_layer = nn.Linear(input_size + hidden_size, 1)
        self.decay_activ = nn.Softplus(beta=4.0)
        self.intensity_layer = nn.Linear(hidden_size, input_size, bias=False)
        self.intensity_activ = nn.Softplus(beta=4.0)

    def forward(self, dt: Tensor, seq_types: Tensor, hidden_ti: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the network.

        Computes the network parameters for the next interval :math:`(t_i,t_{i+1}]` from:

        * the decayed hidden state :math:`h(t_i) = h_i\exp(-\delta_i(t_i-t_{i-1}))` computed
          for the interval :math:`(t_{i-1},t_i]`
        * the interval of time :math:`\Delta t_{i+1} = t_{i+1}-t_i` before the next event

        Args:
            dt: interval of time before event :math:`i`, :math:`\Delta t_i`
                Shape: batch
            seq_types: sequence event types (one hot encoded)
                Shape: batch * input_size
            hidden_ti: decayed hidden state :math:`h(t_i)` at the end of :math:`(t_{i-1},t_i]`
                Shape: batch * hidden_size

        Returns:
            The hidden state and decay value for the interval, and the decayed hidden state.
            Collect them during training to use for computing the loss.
        """
        # seq_types = self.embed(seq_types)
        concat = torch.cat((seq_types, hidden_ti), dim=1)
        # Decay value for the next interval, predicted from the decayed hidden state
        decay = self.decay_activ(self.decay_layer(concat))
        # New hidden state h(t_i+)
        hidden = self.rnn_layer(seq_types, hidden_ti)  # shape batch * hidden_size
        # decay the new hidden state to its value h(t_{i+1})
        dt = dt.unsqueeze(1)
        hidden_after_decay = hidden * torch.exp(-decay * dt)  # shape batch * hidden_size
        return hidden, decay, hidden_after_decay

    def initialize_hidden(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """

        Returns:
            Shape: batch * hidden_size, batch * 1
        """
        return (torch.zeros(batch_size, self.hidden_size, requires_grad=True),
                torch.zeros(batch_size, 1, requires_grad=True))

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
            seq_types: event types
                Shape: N * batch * (K + 1)
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
        # import pdb; pdb.set_trace()
        # get the intensities of the types which are relevant to each event
        # multiplying by the one-hot seq_types tensor sets the non-relevant intensities to 0
        intensity_ev_times_filtered = intensity_ev_times*seq_types.float()
        # reduce on the type dim. (dropping the 0s in the process), then
        # reduce the log-intensities on sequence dim.
        # shape (batch_size,)
        first_term = intensity_ev_times_filtered.sum(dim=2).log().sum(dim=0)
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
            intensity_at_samples, batch_first=True, padding_value=0.0)  # shape N * batch * (K + 1)
        total_intens_samples: Tensor = intensity_at_samples.sum(dim=2, keepdim=True)
        integral_estimates: Tensor = dt_sequence*total_intens_samples
        second_term = integral_estimates.sum(dim=0)
        res = (- first_term + second_term).mean()
        return res

    def generate_sequence(self, tmax: float):
        """
        Generate an event sequence on the interval [0, tmax].

        Args:
            tmax: time horizon

        Returns:
            Sequence of event times with corresponding event intensities.
        """
        with torch.no_grad():
            s = torch.zeros(1)
            last_t = 0.
            hidden, decay = self.initialize_hidden()
            event_times = [last_t]  # record sequence start event
            event_types = [self.input_size]  # sequence start event is of type K
            hidd_hist = []
            decay_hist = []
            max_lbda = self.intensity_activ(
                self.intensity_layer(hidden)).sum(dim=1, keepdim=True)
            # import pdb; pdb.set_trace()

            while last_t < tmax:
                u1: Tensor = torch.rand(1)
                # Candidate inter-arrival time the aggregated process
                ds: Tensor = -1./max_lbda*u1.log()
                # candidate future arrival time
                s = s.clone() + ds
                # adaptive sampling: always update the hidden state
                hidden = hidden*torch.exp(-decay*ds)
                intens_candidate = self.intensity_activ(
                    self.intensity_layer(hidden))
                total_intens: Tensor = torch.sum(intens_candidate, dim=1, keepdim=True)
                # rejection sampling
                u2: Tensor = torch.rand(1)
                if u2 <= total_intens/max_lbda:
                    # shape 1 * (K+1)
                    # probability distribution for the types
                    weights: Tensor = intens_candidate/total_intens  # ratios of types intensities to aggregate
                    res = torch.multinomial(weights, 1)
                    k = res.item()
                    if k < self.input_size:
                        # accept
                        x = one_hot_embedding(res[0], self.input_size)
                        concat = torch.cat((x, hidden), dim=1)
                        decay = self.decay_activ(self.decay_layer(concat))
                        hidden = self.rnn_layer(x, hidden)
                        hidd_hist.append(hidden)
                        decay_hist.append(decay)
                        last_t = s.item()
                        event_times.append(last_t)
                        event_types.append(k)
                max_lbda = total_intens.clone()
            event_times = Tensor(event_times).squeeze(0)
            event_types = Tensor(event_types).squeeze(0)
            decay_hist = torch.stack(decay_hist, dim=2).squeeze(0).squeeze(0)
            return event_times, hidd_hist, decay_hist

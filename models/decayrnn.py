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
        self.rnn_layer = nn.RNNCell(input_size, hidden_size, nonlinearity="relu")
        # self.rnn_layer = nn.Sequential(
        #     nn.Linear(input_size+hidden_size, hidden_size), nn.ReLU())
        self.decay_layer = nn.Sequential(
            nn.Linear(input_size + hidden_size, 1),
            nn.Softplus(beta=3.0)
        )
        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size, bias=False),
            nn.Softplus(beta=3.0)
        )

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
        concat = torch.cat((seq_types, hidden_ti), dim=1)
        # Decay value for the next interval, predicted from the decayed hidden state
        decay = self.decay_layer(concat)
        # New hidden state h(t_i+)
        hidden = self.rnn_layer(seq_types, hidden_ti)  # shape batch * hidden_size
        # decay the new hidden state to its value h(t_{i+1})
        hidden_after_decay = hidden * torch.exp(-decay * dt[:, None])  # shape batch * hidden_size
        return hidden, decay, hidden_after_decay

    def initialize_hidden(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """

        Returns:
            Shape: batch * hidden_size, batch * 1
        """
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, 1))

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
        return self.intensity_layer(h_t)

    def compute_loss(self, seq_times: Tensor, seq_types: Tensor, batch_sizes: Tensor,
                     hiddens: List[Tensor], decays: List[Tensor], tmax: float) -> Tensor:
        """
        Negative log-likelihood

        .. math::
            \mathcal{L} = \log P(\{(t_i, k_i)\}_i)

        There are :math:`N+1` inter-event intervals for the process,
        counting :math:`[0, t_1)` and :math:`(t_N, tmax)`.

        Args:
            seq_times: event sequence, including start time 0
                Shape: (N + 1) * batch
            seq_types: event types, one-hot encoded
                Shape: N * batch * (K + 1)
            batch_sizes: batch sizes for each event sequence tensor, by length
            hiddens:
                Shape: (N + 1) * batch * hidden_size
            decays:
                Shape: N + 1
            tmax: time interval bound

        Returns:

        """
        dt_sequence: Tensor = seq_times[1:] - seq_times[:-1]  # shape N * batch
        n_times = len(hiddens)
        intens_ev_times: Tensor = [
            self.intensity_layer(hiddens[i])
            for i in range(n_times)
        ]
        # shape N * batch * input_dim
        intens_ev_times = nn.utils.rnn.pad_sequence(
            intens_ev_times, batch_first=True, padding_value=1.0)
        # import pdb; pdb.set_trace()
        # get the intensities of the types which are relevant to each event
        # multiplying by the one-hot seq_types tensor sets the non-relevant intensities to 0
        intens_ev_times_filtered = (intens_ev_times*seq_types[:-1]).sum(dim=2)
        # reduce on the type dim. (dropping the 0s in the process), then
        # reduce the log-intensities on seq_times dim.
        # shape (batch_size,)
        first_term = intens_ev_times_filtered.log().sum(dim=0)
        # Take uniform time samples inside of each inter-event interval
        # seq_times: Tensor = torch.cat((seq_times, tmax*torch.ones_like(seq_times[-1:, :])))
        # dt_sequence = seq_times[1:] - seq_times[:-1]
        time_samples = seq_times[:-1] + dt_sequence * torch.rand_like(dt_sequence)  # shape N
        intens_at_samples = [
            self.compute_intensity(hiddens[i], decays[i],
                                   time_samples[i, :batch_sizes[i]], seq_times[i, :batch_sizes[i]])
            for i in range(n_times)
        ]
        intens_at_samples = nn.utils.rnn.pad_sequence(
            intens_at_samples, batch_first=True, padding_value=0.0)  # shape N * batch * (K + 1)
        total_intens_samples: Tensor = intens_at_samples.sum(dim=2, keepdim=True)
        integral_estimates: Tensor = dt_sequence*total_intens_samples
        second_term = integral_estimates.sum(dim=0)
        res = (- first_term + second_term).mean()
        return res


def generate_sequence(model: HawkesDecayRNN, tmax: float):
    """
    Generate an event sequence on the interval [0, tmax].

    Args:
        model: instance of Decay-RNN model
        tmax: time horizon

    Returns:
        Sequence of event times with corresponding event intensities.
    """
    with torch.no_grad():
        s = torch.zeros(1)
        last_t = 0.
        hidden, decay = model.initialize_hidden()
        event_times = [last_t]  # record sequence start event
        event_types = [model.input_size-1]  # sequence start event is of type K
        hidd_hist = []
        decay_hist = []
        max_lbda = model.intensity_layer(hidden).sum(dim=1, keepdim=True)
        # import pdb; pdb.set_trace()

        while last_t < tmax:
            u1: Tensor = torch.rand(1)
            # Candidate inter-arrival time the aggregated process
            ds: Tensor = -1./max_lbda*u1.log()
            # candidate future arrival time
            s = s.clone() + ds
            # adaptive sampling: always update the hidden state
            hidden = hidden*torch.exp(-decay*ds)
            intens_candidate = model.intensity_layer(hidden)
            total_intens: Tensor = torch.sum(intens_candidate, dim=1, keepdim=True)
            # rejection sampling
            u2: Tensor = torch.rand(1)
            if u2 <= total_intens/max_lbda:
                # shape 1 * (K+1)
                # probability distribution for the types
                weights: Tensor = intens_candidate/total_intens  # ratios of types intensities to aggregate
                res = torch.multinomial(weights, 1)
                k = res.item()
                if k < model.input_size:
                    # accept
                    x = one_hot_embedding(res[0], model.input_size)
                    concat = torch.cat((x, hidden), dim=1)
                    decay = model.decay_layer(concat)
                    hidden = model.rnn_layer(x, hidden)
                    hidd_hist.append(hidden)
                    decay_hist.append(decay)
                    last_t = s.item()
                    event_times.append(last_t)
                    event_types.append(k)
            max_lbda = total_intens.clone()
        event_times = Tensor(event_times).squeeze(0)
        event_types = Tensor(event_types).squeeze(0)
        decay_hist = torch.stack(decay_hist, dim=2).squeeze(0).squeeze(0)
        return event_times, event_types, hidd_hist, decay_hist


def read_predict(model: HawkesDecayRNN, event_seq_times: Tensor,
                 event_seq_types: Tensor, seq_length: Tensor,
                 verbose: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Reads an event sequence and predicts the next event time and type.

    Args:
        model: Decay-RNN model instance
        event_seq_times: event sequence arrival times (with 0)
        event_seq_types: event types (one-hot encoded)
        seq_length: event sequence length
        verbose: whether or not to print stuff

    Returns:

    """
    ndim = event_seq_times.ndimension()
    if ndim < 2:
        event_seq_times = event_seq_times.unsqueeze(1)
        event_seq_types = event_seq_types.unsqueeze(1)
    event_seq_times = event_seq_times[:seq_length+1]
    event_seq_types = event_seq_types[:seq_length+1]
    model.eval()
    hidden_t, decay = model.initialize_hidden()
    hidden = hidden_t.clone()
    dt_seq = event_seq_times[1:] - event_seq_times[:-1]
    assert seq_length == dt_seq.shape[0]
    # Read event sequence
    for i in range(seq_length):
        hidden, decay, hidden_t = model(dt_seq[i], event_seq_types[i], hidden_t)
    # We read the types of all events up until this one
    last_ev_time = event_seq_times[-2]  # last read event time
    type_real = event_seq_types[-1]  # real next event's type
    ds = dt_seq[-1]  # time until next event
    with torch.no_grad():
        intensities = model.intensity_layer(hidden)
        # probability distribution of all possible evt types at tN
        prob_distrib = intensities/intensities.sum()
        k_type_predict = torch.multinomial(prob_distrib, 1)[0]  # event type prediction
        type_predict = one_hot_embedding(k_type_predict, model.input_size)
    # import pdb; pdb.set_trace()
    k_type_real = torch.argmax(type_real)
    if verbose:
        print("Sequence length: {}".format(seq_length))
        print("Last read event time: {} of type {}"
              .format(last_ev_time, event_seq_types[-2].argmax()))
        print("Next event time: {} in {} secs".format(event_seq_times[-1], ds))
        print("Actual type: {}".format(k_type_real))
        print("Predicted type: {}".format(k_type_predict.item()))

    return type_real, type_predict, prob_distrib

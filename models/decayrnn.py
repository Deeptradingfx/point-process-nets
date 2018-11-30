import numpy as np
import torch
from torch import nn
from torch import Tensor
from typing import Tuple, List
from utils.load_synth_data import one_hot_embedding


class HawkesDecayRNN(nn.Module):
    """
    Recurrent neural network (RNN) model using decaying hidden states between events, Decay-RNN.

    .. math::
        h(t) = h_i e^{-\delta_i(t-t_i)},\quad t\in (t_{i-1}, t_i]
    """

    def __init__(self, input_size: int, hidden_size: int, intens_bias: bool = False):
        """
        Args:
            input_size: process dimension K
            hidden_size: hidden layer dimension
        """
        super(HawkesDecayRNN, self).__init__()
        self.trained_epochs = 0
        self.process_dim = input_size  # real process dimension
        input_size += 1  # add the dimension of the beginning-of-sequence event type
        self.input_size = input_size  # input size of the embedding layer
        self.hidden_size = hidden_size  # hidden dimension size
        self.embed = nn.Embedding(self.input_size, self.process_dim, padding_idx=self.process_dim)
        self.rnn_layer = nn.RNNCell(self.process_dim, hidden_size, nonlinearity="tanh")
        self.decay_layer = nn.Sequential(
            nn.Linear(self.process_dim + hidden_size, 1),
            nn.Softplus(beta=3.0))
        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_size, self.process_dim, bias=intens_bias),
            nn.Softplus(beta=3.0))

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
        x = self.embed(seq_types)
        concat = torch.cat((x, hidden_ti), dim=1)
        # Decay value for the next interval, predicted from the decayed hidden state
        decay = self.decay_layer(concat)
        # New hidden state h(t_i+)
        hidden = self.rnn_layer(x, hidden_ti)  # shape batch * hidden_size
        # decay the new hidden state to its value h(t_{i+1})
        hidden_after_decay = hidden * torch.exp(-decay * dt[:, None])  # shape batch * hidden_size
        return hidden, decay, hidden_after_decay

    def initialize_hidden(self, batch_size: int = 1, device=None) -> Tuple[Tensor, Tensor]:
        """

        Returns:
            Shape: batch * hidden_size, batch * 1
        """
        (h0, d0) = (torch.zeros(batch_size, self.hidden_size),
                    torch.zeros(batch_size, 1))
        if device:
            h0 = h0.to(device)
            d0 = d0.to(device)
        return h0, d0

    def compute_intensity(self, hidden: Tensor, decay: Tensor, dt: Tensor) -> Tensor:
        """
        Compute the process intensity for the given parameters at the given time:

        .. math::
            h(t) = h_i \exp(-\delta (t - t_{i-1} ))

        Args:
            hidden: hidden state :math:`h_i` at the beginning of the interval
            dt: elapsed time since previous event
            decay: intensity decay :math:`\delta` on interval :math:`[t, \infty)`

        Returns:
            Intensity function value after dt.
        """
        h_t: Tensor = hidden*torch.exp(-decay*dt)
        if h_t.ndimension() > 2:
            h_t = h_t.transpose(1, 2)
        lbda_t: Tensor = self.intensity_layer(h_t)
        if h_t.ndimension() > 2:
            lbda_t = lbda_t.transpose(1, 2)
        return lbda_t

    def compute_loss(self, seq_times: Tensor, seq_onehot_types: Tensor, batch_sizes: Tensor, hiddens: List[Tensor],
                     hiddens_decayed: List[Tensor], decays: List[Tensor], tmax: float) -> Tensor:
        """
        Negative log-likelihood

        .. math::
            \mathcal{L} = \log P(\{(t_i, k_i)\}_i)

        There are :math:`N+1` inter-event intervals for the process,
        counting :math:`[0, t_1)` and :math:`(t_N, tmax)`.

        Args:
            seq_times: event sequence, including start time 0.
                Shape: batch * (N + 1)
            seq_onehot_types: event types, one-hot encoded.
                Shape: batch * N * (K + 1)
            batch_sizes: batch sizes for each event sequence tensor, by length.
            hiddens:
                Shape: batch * (N + 1) * hidden_size
            hiddens_decayed: decayed hidden states.
                Shape: batch * (N + 1) * hidden_size
            decays:
                Shape: batch * (N + 1)
            tmax: time interval bound.

        Returns:

        """
        n_batch = seq_times.size(0)
        n_times = seq_times.size(1) - 1
        # import pdb; pdb.set_trace()
        dt_seq: Tensor = seq_times[:, 1:] - seq_times[:, :-1]  # shape N * batch
        device = seq_times.device
        intens_at_evs: Tensor = [
            self.intensity_layer(hiddens_decayed[i])
            for i in range(1, n_times)  # do not count the 0-th or End-of-sequence events
        ]  # intensities just before the events occur
        # shape batch * N * input_dim
        intens_at_evs = nn.utils.rnn.pad_sequence(
            intens_at_evs, padding_value=1.0)  # pad with 0 to get rid of the non-events
        log_intensities = intens_at_evs.log()  # log intensities
        # get the intensities of the types which are relevant to each event
        # multiplying by the one-hot seq_types tensor sets the non-relevant intensities to 0
        intens_ev_times_filtered = (log_intensities * seq_onehot_types[:, 1:-1]).sum(dim=2)
        # reduce on the type dim. (dropping the 0s in the process), then
        # reduce the log-intensities on seq_times dim.
        # shape (batch_size,)
        first_term = intens_ev_times_filtered.sum(dim=1)
        # Take uniform time samples inside of each inter-event interval
        # seq_times: Tensor = torch.cat((seq_times, tmax*torch.ones_like(seq_times[-1:, :])))
        # dt_sequence = seq_times[1:] - seq_times[:-1]
        n_mc_samples = 10
        # shape N * batch * M_mc
        taus = torch.rand(n_batch, n_times, n_mc_samples).to(device)
        taus: Tensor = dt_seq.unsqueeze(-1) * taus  # inter-event times samples
        intens_at_samples = []
        for i in range(n_times):
            # print(i)
            # print("batchsize", batch_sizes[i])
            # print("next one", batch_sizes[i+1], "with hidden", hiddens[i+1].shape)
            v = self.compute_intensity(hiddens[i].unsqueeze(-1), decays[i].unsqueeze(-1),
                                       taus[:batch_sizes[i], i].unsqueeze(1))
            intens_at_samples.append(v)
        intens_at_samples: Tensor = nn.utils.rnn.pad_sequence(
            intens_at_samples, padding_value=0.0)  # shape batch * N * K * MC
        total_intens_samples: Tensor = intens_at_samples.sum(dim=2)  # shape batch * N * MC
        integral_estimates: Tensor = torch.sum(dt_seq[:, :, None]*total_intens_samples, dim=1)
        second_term: Tensor = integral_estimates.mean(dim=1)
        res: Tensor = (- first_term + second_term).mean()  # average over the bath
        return res


class Generator:
    """
    Event sequence generator for the Hawkes Decay-RNN model.

    Attributes
        model: model instance with which to generate event sequences.
    """

    def __init__(self, model: HawkesDecayRNN):
        self.model = model
        self.dim_process = model.input_size - 1  # process dimension
        print("Process model dim:\t{}\tHidden units:\t{}".format(self.dim_process, model.hidden_size))
        self.event_times = []
        self.event_types = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self.all_times_ = []

    def get_max_lbda(self, hidden):
        partial_lbda = self.model.intensity_layer[0](hidden)
        positive_comps = torch.max(partial_lbda, torch.zeros_like(partial_lbda))
        softmaxed = self.model.intensity_layer[1](positive_comps)
        return softmaxed.sum(dim=1, keepdim=True)

    def restart_sequence(self):
        self.event_times = []
        self.event_types = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self.all_times_ = []

    def generate_sequence(self, tmax: float):
        """
        Generate an event sequence on the interval [0, tmax].

        Args:
            tmax: time horizon

        Returns:
            Sequence of event times with corresponding event intensities.
        """
        self.restart_sequence()
        model = self.model
        with torch.no_grad():
            s = torch.zeros(1)
            last_t = 0.
            hidden, decay = model.initialize_hidden()
            self.event_times.append(last_t)  # record sequence start event
            self.event_types.append(self.dim_process)  # sequence start event is of type K
            max_lbda = self.get_max_lbda(hidden)
            # import pdb; pdb.set_trace()
            while last_t < tmax:
                u1: Tensor = torch.rand(1)
                # Candidate inter-arrival time the aggregated process
                ds: Tensor = -1./max_lbda*u1.log()
                # candidate future arrival time
                du = ds.item()/10
                u = s.item() + du
                s: Tensor = s + ds
                if s > tmax:
                    break
                # Track event intensities
                h_u = hidden.clone()
                while u < s.item():
                    self.all_times_.append(u)
                    h_u = h_u*torch.exp(-decay*du)
                    lbda_t = model.intensity_layer(h_u)
                    self.intens_hist.append(lbda_t)
                    u += du
                self.all_times_.append(s.item())
                # adaptive sampling: always update the hidden state
                hidden = hidden*torch.exp(-decay*ds)
                intens_candidate = model.intensity_layer(hidden)
                self.intens_hist.append(intens_candidate)
                total_intens: Tensor = torch.sum(intens_candidate, dim=1, keepdim=True)
                # rejection sampling
                u2: Tensor = torch.rand(1)
                ratio = total_intens/max_lbda
                if u2 <= ratio:
                    # shape 1 * (K+1)
                    # probability distribution for the types
                    weights: Tensor = intens_candidate/total_intens  # ratios of types intensities to aggregate
                    res = torch.multinomial(weights, 1)
                    k = res.item()
                    # accept
                    x = model.embed(res[0])
                    concat = torch.cat((x, hidden), dim=1)
                    decay = model.decay_layer(concat)
                    hidden = model.rnn_layer(x, hidden)
                    self.hidden_hist.append(hidden)
                    self.decay_hist.append(decay)
                    last_t = s.item()
                    self.event_times.append(last_t)
                    self.event_types.append(k)
                max_lbda = self.get_max_lbda(hidden)


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

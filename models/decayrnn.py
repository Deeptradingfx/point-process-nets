import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
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
            nn.Linear(self.process_dim + hidden_size, hidden_size),
            nn.Softplus(beta=3.0))
        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_size, self.process_dim, bias=intens_bias),
            nn.Softplus(beta=3.0))

    def forward(self, dt: PackedSequence, seq_types: PackedSequence,
                h0: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Forward pass of the network.

        Computes the network parameters for the next interval :math:`(t_i,t_{i+1}]` from:

        * the decayed hidden state :math:`h(t_i) = h_i\exp(-\delta_i(t_i-t_{i-1}))` computed
          for the interval :math:`(t_{i-1},t_i]`
        * the interval of time :math:`\Delta t_{i+1} = t_{i+1}-t_i` before the next event

        Args:
            dt: interval of time before event :math:`i`, :math:`\Delta t_i`
                Shape: seq_len * batch
            seq_types: sequence event types (one hot encoded)
                Shape: seq_len * batch * input_size
            h0: decayed hidden state :math:`h(t_i)` at the end of :math:`(t_{i-1},t_i]`
                Shape: batch * hidden_size

        Returns:
            The hidden state and decay value for the interval, and the decayed hidden state.
            Collect them during training to use for computing the loss.
        """
        h_t = h0
        hiddens = []  # full, updated hidden states
        hiddens_ti = []  # decayed hidden states
        decays = []  # decay parameters
        max_seq_length = len(dt.batch_sizes)
        beg_index = 0
        for j in range(max_seq_length):
            # event t_i is happening
            batch_size = dt.batch_sizes[j]
            # hidden state just before this event
            h_t = h_t[:batch_size]
            # time until next event t_{i+1}
            dt_sub_batch = dt.data[beg_index:(beg_index + batch_size)]
            types_sub_batch = seq_types.data[beg_index:(beg_index + batch_size)]

            x = self.embed(types_sub_batch)
            concat = torch.cat((x, h_t), dim=1)
            # Decay value for the next interval, predicted from the decayed hidden state
            decay = self.decay_layer(concat)
            # New hidden state h(t_i+)
            hidden = self.rnn_layer(x, h_t)  # shape batch * hidden_size
            # decay the new hidden state to its value h(t_{i+1})
            h_t = hidden * torch.exp(-decay * dt_sub_batch[:, None])  # shape batch * hidden_size

            beg_index += batch_size
            hiddens.append(hidden)
            hiddens_ti.append(h_t)
            decays.append(decay)
        return hiddens, decays, hiddens_ti

    def init_hidden(self, batch_size: int = 1, device=None) -> Tensor:
        """

        Returns:
            Shape: batch * hidden_size, batch * 1
        """
        h0 = torch.zeros(batch_size, self.hidden_size)
        if device:
            h0 = h0.to(device)
        return h0

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
        h_t: Tensor = hidden * torch.exp(-decay * dt)
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
        n_times = len(batch_sizes)
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
        log_sum = intens_ev_times_filtered.sum(dim=1)
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
        partial_integrals: Tensor = dt_seq * total_intens_samples.mean(dim=2)
        integral_: Tensor = partial_integrals.sum(dim=1)
        res: Tensor = (- log_sum + integral_).mean()  # average over the bath
        return res


class HawkesRNNGen:
    """
    Event sequence generator for the Hawkes Decay-RNN model.

    Attributes
        model: model instance with which to generate event sequences.
    """

    def __init__(self, model: HawkesDecayRNN, record_intensity: bool = True):
        self.model = model
        self.process_dim = model.input_size - 1  # process dimension
        print("Process model dim:\t{}\tHidden units:\t{}".format(self.process_dim, model.hidden_size))
        self.event_times = []
        self.event_types = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self._plot_times = []
        self.event_intens = []
        self.record_intensity: bool = record_intensity

    def get_max_lbda(self, hidden):
        partial_lbda = self.model.intensity_layer[0](hidden)
        positive_comps = torch.max(partial_lbda, torch.zeros_like(partial_lbda))
        softmaxed = self.model.intensity_layer[1](positive_comps)
        return softmaxed.sum(dim=1, keepdim=True)

    def restart_sequence(self):
        self.event_times = []
        self.event_types = []
        self.event_intens = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self._plot_times = []

    def generate_sequence(self, tmax: float, record_intensity: bool = None,
                          mult_ub: float = 20.0):
        """
        Generate an event sequence on the interval [0, tmax].

        Args:
            tmax: time horizon
            record_intensity (bool): whether or not to record the intensity at every point.
            mult_ub: multiplication factor for the rejection sampling upper bound, useful for plotting.

        Returns:
            Sequence of event times with corresponding event intensities.
        """
        self.restart_sequence()
        if record_intensity is None:
            record_intensity = self.record_intensity
        else:
            self.record_intensity = record_intensity
        if not record_intensity:
            mult_ub = 1.
        model = self.model
        with torch.no_grad():
            s = torch.zeros(1)
            last_t = 0.
            hidden = model.init_hidden()
            hidden.normal_(std=0.1)  # set the hidden state to a N(0, 0.01) variable.
            decay = torch.zeros_like(hidden)
            intens = model.intensity_layer(hidden).numpy()
            self.hidden_hist.append(hidden.numpy())
            self.event_times.append(last_t)  # record sequence start event
            self.event_types.append(self.process_dim)  # sequence start event is of type K
            self.event_intens.append(intens)
            self.intens_hist.append(intens)
            self._plot_times.append(last_t)
            max_lbda = mult_ub*self.get_max_lbda(hidden)
            # import pdb; pdb.set_trace()
            while last_t <= tmax:
                u1: Tensor = torch.rand(1)
                # Candidate inter-arrival time the aggregated process
                ds: Tensor = -1. / max_lbda * u1.log()
                # candidate future arrival time
                s: Tensor = s + ds
                if s > tmax:
                    break
                self._plot_times.append(s.item())
                # adaptive sampling: always update the hidden state
                hidden = hidden * torch.exp(-decay * ds)
                intens_candidate = model.intensity_layer(hidden)
                self.intens_hist.append(intens_candidate.numpy())
                total_intens: Tensor = torch.sum(intens_candidate, dim=1, keepdim=True)
                # rejection sampling
                u2: Tensor = torch.rand(1)
                ratio = total_intens / max_lbda
                if u2 <= ratio:
                    # shape 1 * (K+1)
                    # probability distribution for the types
                    weights: Tensor = intens_candidate / total_intens  # ratios of types intensities to aggregate
                    res = torch.multinomial(weights, 1)
                    k = res.item()
                    # accept
                    x = model.embed(res[0])
                    concat = torch.cat((x, hidden), dim=1)
                    decay = model.decay_layer(concat)
                    hidden = model.rnn_layer(x, hidden)
                    self.hidden_hist.append(hidden.numpy())
                    self.decay_hist.append(decay.numpy())
                    last_t = s.item()
                    self.event_times.append(last_t)
                    self.event_types.append(k)
                    self._plot_times.append(last_t)
                    intens = model.intensity_layer(hidden).numpy()
                    self.event_intens.append(intens)
                    self.intens_hist.append(intens)
                max_lbda = mult_ub*self.get_max_lbda(hidden)

    def plot_events_and_intensity(self, model_name: str = None, debug=False):
        assert self.record_intensity
        import matplotlib.pyplot as plt
        gen_seq_times = self.event_times
        gen_seq_types = self.event_types
        sequence_length = len(gen_seq_times)
        print("no. of events: {}".format(sequence_length))
        evt_times = np.array(gen_seq_times)
        evt_types = np.array(gen_seq_types)
        fig, ax = plt.subplots(1, 1, sharex='all', dpi=100,
                               figsize=(10, 4.5))
        ax: plt.Axes
        inpt_size = self.process_dim
        ax.set_xlabel('Time $t$ (s)')
        intens_hist = np.stack(self.intens_hist)[:, 0]
        labels = ["type {}".format(i) for i in range(self.process_dim)]
        for y, lab in zip(intens_hist.T, labels):
            ax.plot(self._plot_times, y, linewidth=.7, label=lab)
        ax.set_ylabel(r"Intensities $\lambda^i_t$")
        title = "Event arrival times and intensities for generated sequence"
        if model_name is None:
            model_name = self.model.__class__.__name__
        title += " ({})".format(model_name)
        ax.set_title(title)
        ylims = ax.get_ylim()
        ts_y = np.stack(self.event_intens)[:, 0]
        for k in range(inpt_size):
            mask = evt_types == k
            print(k, end=': ')
            if k == self.process_dim:
                print("starter type")
                # label = "start event".format(k)
                y = self.intens_hist[0].sum(axis=1)
            else:
                print("type {}".format(k))
                y = ts_y[mask, k]
                # label = "type {} event".format(k)
            ax.scatter(evt_times[mask], y, s=9, zorder=5,
                       alpha=0.8)
            ax.vlines(evt_times[mask], ylims[0], ylims[1], linewidth=0.3, linestyles='-', alpha=0.8)

        # Useful for debugging the sampling for the intensity curve.
        if debug:
            for s in self._plot_times:
                ax.vlines(s, ylims[0], ylims[1], linewidth=0.3, linestyles='--', alpha=0.6, colors='red')

        ax.set_ylim(*ylims)
        ax.legend()
        fig.tight_layout()
        return fig


def read_predict(model: HawkesDecayRNN, seq_times: Tensor,
                 seq_types: Tensor, seq_length: Tensor,
                 verbose: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Reads an event sequence and predicts the next event time and type.

    Args:
        model: Decay-RNN model instance
        seq_times: event sequence arrival times (with 0)
        seq_types: event types (one-hot encoded)
        seq_length: event sequence length
        verbose: whether or not to print stuff

    Returns:

    """
    ndim = seq_times.ndimension()
    if ndim < 2:
        seq_times = seq_times.unsqueeze(1)
        seq_types = seq_types.unsqueeze(1)
    seq_times = seq_times[:seq_length + 1]
    seq_types = seq_types[:seq_length + 1]
    model.eval()
    hidden_t, decay = model.init_hidden()
    hidden = hidden_t.clone()
    dt_seq = seq_times[1:] - seq_times[:-1]
    assert seq_length == dt_seq.shape[0]
    # Read event sequence
    for i in range(seq_length):
        hidden, decay, hidden_t = model(dt_seq[i], seq_types[i], hidden_t)
    # We read the types of all events up until this one
    last_ev_time = seq_times[-2]  # last read event time
    type_real = seq_types[-1]  # real next event's type
    ds = dt_seq[-1]  # time until next event
    with torch.no_grad():
        intensities = model.intensity_layer(hidden)
        # probability distribution of all possible evt types at tN
        prob_distrib = intensities / intensities.sum()
        k_type_predict = torch.multinomial(prob_distrib, 1)[0]  # event type prediction
        type_predict = one_hot_embedding(k_type_predict, model.input_size)
    # import pdb; pdb.set_trace()
    k_type_real = torch.argmax(type_real)
    if verbose:
        print("Sequence length: {}".format(seq_length))
        print("Last read event time: {} of type {}"
              .format(last_ev_time, seq_types[-2].argmax()))
        print("Next event time: {} in {} secs".format(seq_times[-1], ds))
        print("Actual type: {}".format(k_type_real))
        print("Predicted type: {}".format(k_type_predict.item()))

    return type_real, type_predict, prob_distrib

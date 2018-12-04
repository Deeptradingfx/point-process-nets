"""
Neural network models for point processes.

@author: manifold
"""
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple, List


class HawkesLSTMCell(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int):
        super(HawkesLSTMCell, self).__init__()
        self.input_g = nn.Linear(input_dim + hidden_size, hidden_size)
        self.forget_g = nn.Linear(input_dim + hidden_size, hidden_size)
        self.output_g = nn.Linear(input_dim + hidden_size, hidden_size)
        self.input_target = nn.Linear(input_dim + hidden_size, hidden_size)
        self.forget_target = nn.Linear(input_dim + hidden_size, hidden_size)
        # activation will be tanh
        self.z_gate = nn.Linear(input_dim + hidden_size, hidden_size)
        # Cell decay factor, identical for all hidden dims
        self.decay_layer = nn.Sequential(
            nn.Linear(input_dim + hidden_size, hidden_size),
            nn.Softplus(beta=3.))

    def forward(self, x, h_t, c_t, c_target) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the updated LSTM paramters.

        Args:s
            x: event type embedding
            h_t:
            c_t:
            c_target:

        Returns:
            h_i: just-updated hidden state
            h_t: hidden state just before next event
            cell_i: just-updated cell state
            c_t: cell state decayed to before next event
            c_target_i: cell state target before the next event
            output: LSTM output
            decay_i: rate of decay for the cell state
        """
        v = torch.cat((x, h_t), dim=1)
        inpt = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        input_target = torch.sigmoid(self.input_target(v))
        forget_target = torch.sigmoid(self.forget_target(v))
        output = torch.sigmoid(self.output_g(v))  # compute the LSTM network output
        # Not-quite-c
        z_i = torch.tanh(self.z_gate(v))
        # Compute the decay parameter
        decay = self.decay_layer(v)
        # Update the cell state to c(t_i+)
        c_i = forget * c_t + inpt * z_i
        # h_i = output * torch.tanh(c_i)  # hidden state just after event
        # Update the cell state target
        c_target = forget_target * c_target + input_target * z_i
        # Decay the cell state to its value before the known next event at t+dt
        # used for the next pass in the loop
        return c_i, c_target, output, decay


class HawkesLSTM(nn.Module):
    """
    A continuous-time LSTM, defined according to Eisner & Mei's article
    https://arxiv.org/abs/1612.09328
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(HawkesLSTM, self).__init__()
        self.process_dim = input_size
        self.trained_epochs = 0
        input_size += 1
        self.input_size = input_size  # embedding input size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, self.process_dim, padding_idx=self.process_dim)
        self.lstm_cell = HawkesLSTMCell(self.process_dim, hidden_size)
        # activation for the intensity
        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_size, self.process_dim, bias=False),  # no bias in the model
            nn.Softplus(beta=3.))

    def init_hidden(self, batch_size: int = 1, device=None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Initialize the hidden state and the cell state.
        The initial cell state target is equal to the initial cell state.
        The first dimension is the batch size.

        Returns:
            (hidden, cell_state)
        """
        (h0, c0, c_target0) = (torch.zeros(batch_size, self.hidden_size),
                               torch.zeros(batch_size, self.hidden_size),
                               torch.zeros(batch_size, self.hidden_size))
        if device:
            h0 = h0.to(device)
            c0 = c0.to(device)
            c_target0 = c_target0.to(device)
        return h0, c0, c_target0

    def forward(self, seq_dt: PackedSequence, seq_types: PackedSequence,
                h0: Tensor, c0: Tensor, c_target0: Tensor
                ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Forward pass of the LSTM network.

        Computes the network parameters for the next interval :math:`(t_i,t_{i+1}]` from the input
        at time :math:`t_{i}`.

        Args:
            seq_dt (PackedSequence): time until next event
                Shape: seq_len * batch
            seq_types (PackedSequence): event sequence types
                Shape: seq_len * batch * input_size
            h0 (Tensor): initial hidden state
            c0 (Tensor): initial cell state
            c_target0 (Tensor): initial target cell state

        Returns:
            The new hidden states and LSTM output.
            outputs: computed by the output gates
            hidden_ti: decayed hidden states hidden state
            cells: cell states
            cell_targets: target cell states
            decays: decay parameters for each interval
        """
        h_t = h0  # continuous hidden state
        c_t = c0  # continuous cell state
        c_target = c_target0  # cell state target
        hiddens_ti = []  # decayed hidden states, directly used in log-likelihood computation
        outputs = []  # output from each LSTM pass
        cells = []  # cell states at event times
        cell_targets = []  # target cell states for each interval
        decays = []  # decays computed at each event
        max_seq_length = len(seq_dt.batch_sizes)
        beg_index = 0
        # loop over all events
        for i in range(max_seq_length):
            batch_size = seq_dt.batch_sizes[i]
            h_t = h_t[:batch_size]
            c_t = c_t[:batch_size]
            c_target = c_target[:batch_size]
            dt = seq_dt.data[beg_index:(beg_index + batch_size)]
            types_sub_batch = seq_types.data[beg_index:(beg_index + batch_size)]

            # Update the hidden states and LSTM parameters following the equations
            x = self.embed(types_sub_batch)
            cell_i, c_target, output, decay_i = self.lstm_cell(x, h_t, c_t, c_target)
            c_t: Tensor = (
                    c_target + (cell_i - c_target) *
                    torch.exp(-decay_i * dt[:, None])
            )
            h_t: Tensor = output * torch.tanh(c_t)  # decayed hidden state just before next event

            outputs.append(output)  # record it
            decays.append(decay_i)
            cells.append(cell_i)  # record it
            cell_targets.append(c_target)  # record
            hiddens_ti.append(h_t)  # record it
        return hiddens_ti, outputs, cells, cell_targets, decays

    def compute_loss(self, seq_times: Tensor, seq_onehot_types: Tensor, batch_sizes: Tensor, hiddens_ti: List[Tensor],
                     cells: List[Tensor], cell_targets: List[Tensor], outputs: List[Tensor],
                     decays: List[Tensor], tmax: float) -> Tensor:
        """
        Compute the negative log-likelihood as a loss function.
        
        Args:
            seq_times: event occurrence timestamps
            seq_onehot_types: types of events in the sequence, one hot encoded
            batch_sizes: batch sizes for each event sequence tensor, by length
            hiddens_ti: hidden states just before the events occur.
            cells: entire cell state history
            cell_targets: cell state target values history
            outputs: entire output history
            decays: entire decay history
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """
        n_batch = seq_times.size(0)
        n_times = len(hiddens_ti)
        dt_seq: Tensor = seq_times[:, 1:] - seq_times[:, :-1]
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs: Tensor = [
            self.intensity_layer(hiddens_ti[i])
            for i in range(1, n_times)  # do not count the 0-th or End-of-sequence events
        ]  # intensities just before the events occur
        # shape batch * N * input_dim
        intens_at_evs = nn.utils.rnn.pad_sequence(
            intens_at_evs, padding_value=1.0)  # pad with 0 to get rid of the non-events
        log_intensities: Tensor = intens_at_evs.log()  # log intensities
        # get the intensities of the types which are relevant to each event
        # multiplying by the one-hot seq_types tensor sets the non-relevant intensities to 0
        log_sum = (log_intensities * seq_onehot_types[:, 1:-1]).sum(dim=(2, 1))  # shape batch

        # COMPUTE INTEGRAL TERM
        # Computed using Monte Carlo method
        # Take uniform time samples inside of each inter-event interval
        # seq_times: Tensor = torch.cat((seq_times, tmax*torch.ones_like(seq_times[-1:, :])))
        # dt_sequence = seq_times[1:] - seq_times[:-1]
        n_mc_samples = 1
        # shape N * batch * M_mc
        taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)
        taus: Tensor = dt_seq[:, :, None, None] * taus  # inter-event times samples)
        intens_at_samples = []
        for i in range(n_times):
            c_target_i = cell_targets[i].unsqueeze(-1)
            c_i = cells[i].unsqueeze(-1)
            c_t = c_target_i + (c_i - c_target_i)*torch.exp(-decays[i].unsqueeze(-1)*taus[:batch_sizes[i], i])
            h_t = outputs[i].unsqueeze(-1) * torch.tanh(c_t)
            h_t = h_t.transpose(1, 2)
            intens_at_samples.append(self.intensity_layer(h_t).transpose(1, 2))
        intens_at_samples = nn.utils.rnn.pad_sequence(
            intens_at_samples, padding_value=0.0)  # shape batch * N * K * MC
        total_intens_samples: Tensor = intens_at_samples.sum(dim=2)  # shape batch * N * MC
        partial_integrals: Tensor = dt_seq * total_intens_samples.mean(dim=2)
        integral_ = partial_integrals.sum(dim=1)
        res = (- log_sum + integral_).mean()  # mean on batch dim
        return res


class HawkesLSTMGen:
    """
    Sequence generator for the CT-LSTM model.
    """

    def __init__(self, model: HawkesLSTM, record_intensity: bool = True):
        self.model = model
        self.process_dim = model.input_size - 1  # process dimension
        print("Process model dim:\t{}\tHidden units:\t{}".format(self.process_dim, model.hidden_size))
        self.event_times = []
        self.event_types = []
        self.event_intens = []
        self.decay_hist: List[Tensor] = []
        self.intens_hist: List[Tensor] = []
        self._plot_times = []  # used for plotting
        self.lbda_ub = []
        self.record_intensity: bool = record_intensity

    def _restart_sequence(self):
        self.event_times = []
        self.event_types = []
        self.event_intens = []
        self.decay_hist = []
        self.intens_hist = []
        self._plot_times = []
        self.lbda_ub = []

    def generate_sequence(self, tmax: float, record_intensity: bool = False):
        """
        Generate an event sequence on the interval [0, tmax].

        Args:
            tmax: maximum time.
            record_intensity (bool): whether or not to record the intensity (e.g. for plotting)
        """
        self._restart_sequence()
        model = self.model
        model.eval()
        if record_intensity is None:
            record_intensity = self.record_intensity
        with torch.no_grad():
            last_t = 0.0
            s = torch.zeros(1)
            h0, c0, _ = model.init_hidden()
            h0.normal_(std=0.1)
            c0.normal_(std=0.1)
            h_t = h0
            c_t = c0
            c_target = c0.clone()
            # Compute the first hidden states from the noise, at t = 0
            k0 = torch.LongTensor([self.process_dim])
            x0 = model.embed(k0)  # the starter event has an embedding of [ 0. ]
            c_t, c_target, output, decay = model.lstm_cell(
                x0, h_t, c_t, c_target
            )
            intens = model.intensity_layer(output * torch.tanh(c_t))
            # Cell state at the last event
            c_t_last = c_t
            # Record everything
            self.event_times.append(last_t)
            self.event_types.append(self.process_dim)
            self.event_intens.append(intens.numpy())
            self.decay_hist.append(decay.numpy())
            self._plot_times.append(last_t)
            self.intens_hist.append(intens.numpy())
            max_lbda = self.update_lbda_bound(output, c_t, c_target).sum()
            self.lbda_ub.append(max_lbda.numpy())
            # max_lbda = 3.0

            while last_t <= tmax:
                # print(max_lbda)
                ds: torch.Tensor = torch.empty_like(s)
                ds.exponential_(max_lbda.item())
                s: Tensor = s + ds.item()  # Increment s
                if s > tmax:
                    break
                time_elapsed_last = s - last_t
                # Compute the current intensity
                # Compute what the cell state at s is
                # Decay the hidden state
                c_t = c_target + (c_t_last - c_target)*torch.exp(-decay*time_elapsed_last)
                h_t = output * torch.tanh(c_t)
                # Apply intensity layer
                intens: Tensor = self.model.intensity_layer(h_t)
                self._plot_times.append(s.item())  # record this time and intensity value
                self.intens_hist.append(intens.numpy())
                u2 = torch.rand(1)  # random in [0,1]
                total_intens = intens.sum(dim=1)
                ratio = total_intens/max_lbda
                if u2 <= ratio:
                    # shape 1 * K
                    # probability distribution for the types
                    weights: Tensor = intens / total_intens  # ratios of types intensities to aggregate
                    k_vector = torch.multinomial(weights, 1)
                    k = k_vector.item()
                    # accept
                    x = model.embed(k_vector[0])
                    # Bump the hidden states
                    c_t_last, c_target, output, decay = model.lstm_cell(
                        x, h_t, c_t, c_target)
                    h_t = output * torch.tanh(c_t_last)
                    intens = model.intensity_layer(h_t)
                    max_lbda = self.update_lbda_bound(output, c_t_last, c_target)
                    max_lbda = 10*max_lbda.sum()
                    self.lbda_ub.append(max_lbda.numpy())
                    # max_lbda = 3.0
                    last_t = s.item()
                    self._plot_times.append(last_t)  # record the time and intensity a second time
                    self.intens_hist.append(intens.numpy())
                    self.decay_hist.append(decay.numpy())
                    self.event_intens.append(intens.numpy())
                    self.event_types.append(k)
                    self.event_times.append(last_t)

    def update_lbda_bound(self, output: Tensor, c_t: Tensor, c_target: Tensor) -> torch.Tensor:
        r"""
        Considering current time is s and knowing the last event, find a new upper bound for the intensities.
        
        Each term in the sum defining ``pre_lambda`` (intensity before activation) is of the form

        .. math::
            \tilde{\lambda}_t =
            w^T o\,\tanh\left(c_{i+1} + (c_{i+1} - \bar c_{i+1}\right)\exp(-\delta(t - t_i)))


        """
        w_alpha = self.model.intensity_layer[0].weight.data
        cell_gap: Tensor = c_t - c_target
        cell_gap = cell_gap.expand_as(w_alpha)
        increasing_mask_ = (cell_gap * w_alpha < 0.0)
        cell_gap[increasing_mask_] = 0.0
        decreasing_mask_ = ~increasing_mask_
        cell_gap[decreasing_mask_] = 1.0
        # upper bound matrix
        upper_bounds: Tensor = w_alpha * output * torch.tanh(c_target + cell_gap)  # shape K * D
        pre_lbda = upper_bounds.sum(dim=1)
        # compute the intensity using the Softplus inside the intensity layer of the model
        res: Tensor = self.model.intensity_layer[1](pre_lbda)
        return res

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


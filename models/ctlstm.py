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


class HawkesLSTM(nn.Module):
    """
    A continuous-time LSTM, defined according to Eisner & Mei's article
    https://arxiv.org/abs/1612.09328
    """

    def __init__(self, input_size: int, hidden_dim: int):
        super(HawkesLSTM, self).__init__()
        self.process_dim = input_size
        self.trained_epochs = 0
        input_size += 1
        self.input_size = input_size  # embedding input size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_size, self.process_dim, padding_idx=self.process_dim)
        self.input_g = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.forget_g = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_g = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.input_target = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.forget_target = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # activation will be tanh
        self.z_gate = nn.Sequential(
            nn.Linear(input_size + hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Cell decay factor, identical for all hidden dims
        self.decay_layer = nn.Sequential(
            nn.Linear(input_size + hidden_dim, 1),
            nn.Softplus(beta=8.)
        )
        # activation for the intensity
        self.activation = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=False),  # no bias in the model
            nn.Softplus(beta=5.)
        )

    def init_hidden(self, batch_size: int = 1, device=None):
        """
        Initialize the hidden state and the cell state.
        The initial cell state target is equal to the initial cell state.
        The first dimension is the batch size.

        Returns:
            (hidden, cell_state)
        """
        (h0, c0) = (torch.zeros(batch_size, self.hidden_dim),
                    torch.zeros(batch_size, self.hidden_dim))
        if device:
            h0 = h0.to(device)
            c0 = c0.to(device)
        return h0, c0

    def forward(self, dt: PackedSequence, seq_types: PackedSequence,
                h0: Tensor, c0: Tensor
                ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Forward pass of the CT-LSTM network.

        Computes the network parameters for the next interval :math:`(t_i,t_{i+1}]` from the input states given by
        at time :math:`t_{i}`, right before the update.

        Args:
            dt: time until next event
                Shape: seq_len * batch
            seq_types: one-hot encoded event sequence types
                Shape: seq_len * batch * input_size
            h0: initial hidden state
            c0: initial cell state

        Returns:
            outputs: computed by the output gates
            hidden_ti: decayed hidden states hidden state
            cells: cell states
            cell_targets: target cell states
            decays: decay parameters for each interval
        """
        h_t = h0  # continuous hidden state
        c_t = c0  # continuous cell state
        c_target_i = c0  # cell state target
        hiddens = []  # full, updated hidden states
        hiddens_ti = []  # decayed hidden states, directly used in log-likelihood computation
        outputs = []  # output from each LSTM pass
        cells = []  # cell states at event times
        cell_targets = []  # target cell states for each interval
        decays = []  # decays computed at each event
        max_seq_length = len(dt.batch_sizes)
        beg_index = 0
        # loop over all events
        for j in range(max_seq_length):
            batch_size = dt.batch_sizes[j]
            h_t = h_t[:batch_size]
            dt_sub_batch = dt.data[beg_index:(beg_index + batch_size)]
            types_sub_batch = seq_types.data[beg_index:(beg_index + batch_size)]

            x = self.embed(types_sub_batch)
            v = torch.cat((x, h_t), dim=1)
            inpt = self.input_g(v)
            forget = self.forget_g(v)
            input_target = self.input_target(v)
            forget_target = self.forget_target(v)
            output = self.output_g(v)
            # Not-quite-c
            z_i = self.z_gate(v)
            # Compute the decay parameter
            decay_i = self.decay_layer(v)
            decays.append(decay_i)
            # Update the cell state to c(t_i+)
            cell_i = forget * c_t + inpt * z_i
            cells.append(cell_i)  # record it

            h_i = output * torch.tanh(cell_i)  # hidden state just after event
            hiddens.append(h_i)  # record it

            # Update the cell state target
            c_target_i = forget_target * c_target_i + input_target * z_i
            cell_targets.append(c_target_i)  # record
            # Decay the cell state to its value before the known next event at t+dt
            # used for the next pass in the loop
            c_t = (
                    c_target_i + (cell_i - c_target_i) *
                    torch.exp(-decay_i * dt_sub_batch[:, None])
            )
            h_t = output * torch.tanh(c_t)  # decayed hidden state just before next event
            hiddens_ti.append(h_t)  # record it
        return hiddens, hiddens_ti, outputs, cells, cell_targets, decays

    def compute_intensity(self, output: Tensor, c_i: Tensor, c_target_i: Tensor, decay: Tensor, dt: Tensor) -> Tensor:
        """
        Compute the intensity at time :math:`t`, given the LSTM parameters on the interval
        :math:`(t_{i-1}, t_i]` and the elapsed time :math:`t-t_{i-1}` since the last
        event.

        Args:
            dt: time elapsed since last event
                Shape: batch
            output: LSTM cell output
                Shape: seq_length * batch * hidden_dim
            c_i: LSTM cell state just after the last event
            c_target_i: cell state target
            decay: decay speed parameter

        Shape:
            batch_size * hidden_dim
        """
        # Get the current continuous-time cell state
        c_t = (
                c_target_i + (c_i - c_target_i) *
                torch.exp(-decay * dt)
        )
        # Compute the hidden state
        h_t = output * torch.tanh(c_t)
        return self.activation(h_t)

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
        dt_seq: Tensor = seq_times[1:] - seq_times[:-1]
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs: Tensor = [
            self.intensity_layer(hiddens_ti[i])
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

        # COMPUTE INTEGRAL TERM
        # Computed using Monte Carlo method

        # Take uniform time samples inside of each inter-event interval
        # seq_times: Tensor = torch.cat((seq_times, tmax*torch.ones_like(seq_times[-1:, :])))
        # dt_sequence = seq_times[1:] - seq_times[:-1]
        n_mc_samples = 10
        # shape N * batch * M_mc
        taus = torch.rand(n_batch, n_times, n_mc_samples).to(device)
        taus: Tensor = dt_seq.unsqueeze(-1) * taus  # inter-event times samples

        intens_at_samples = [
            self.compute_intensity(outputs[i], cells[i], cell_targets[i], decays[i],
                                   taus[i, :batch_sizes[i]])
            for i in range(n_times)
        ]
        intens_at_samples = nn.utils.rnn.pad_sequence(
            intens_at_samples, batch_first=True, padding_value=0.0)  # shape N * batch * (K + 1)
        total_intens_samples: Tensor = intens_at_samples.sum(dim=2, keepdim=True)
        integral_estimates: Tensor = dt_seq * total_intens_samples
        second_term = integral_estimates.sum(dim=0)
        res = (- log_sum + second_term).mean()
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
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self.all_times_ = []
        self.event_intens = []
        self.record_intensity: bool = record_intensity

    def generate_sequence(self, tmax: float) -> List[torch.Tensor]:
        """
        Generate a sequence of events distributed according to the neural Hawkes model.

        WARNING: Check the model is in evaluation mode!
        """
        # Reinitialize sequence
        self.sequence_ = []
        s = 0.0
        self.sequence_.append(s)
        self.update_hidden_state(s)
        lbda_max = self.update_max_lambda()
        self.lbda_max_seq_.append(lbda_max)

        while s <= tmax:
            u1 = np.random.rand()
            dt: torch.Tensor = -1. / lbda_max * np.log(u1)
            s += dt.item()  # Increment s
            print("Increment: {:}".format(dt.item()))
            if s > tmax:
                break
            t = self.sequence_[-1]
            # Compute the current intensity
            # Compute what the hidden state at s is
            cell_state_s = (
                    self.cell_target + (self.cell_t - self.cell_target)
                    * torch.exp(-self.cell_decay * (s - t))
            )
            hidden_state_s = self.output * torch.tanh(cell_state_s)
            # Apply activation function to hidden state
            intens = self.model.activation(hidden_state_s).item()
            u2 = np.random.rand()  # random in [0,1]
            print("Intensity {:}".format(intens))
            print("\tlbda_max\t{:}".format(lbda_max))
            print("\tratio\t{:}".format(intens / lbda_max))
            if u2 <= intens / lbda_max:
                lbda_max = self.update_max_lambda()
                self.lbda_max_seq_.append(lbda_max)
                self.update_hidden_state(s)
                self.sequence_.append(s)
        self.sequence_.pop(0)
        return self.sequence_

    def update_max_lambda(self) -> torch.Tensor:
        """
        Considering current time is s and knowing the last event, find a new maximum value of the intensity.
        
        Each term in the sum defining pre_lambda (intensity before activation) is of the form
        .. math::
            o_k\tanh([c_{i+1}]_k + ([c_{i+1}]_k - [\bar c_{i+1}]_k)\exp(-\delta_k(t - t_i)))
        """
        w_alpha = self.model.activation[0].weight.data
        cell_diff = self.cell_t - self.cell_target
        mult_prefix = w_alpha * self.output
        pos_prefactor = mult_prefix > 0
        pos_decr = pos_prefactor & (cell_diff >= 0)
        p1 = torch.dot(mult_prefix[pos_decr], torch.tanh(self.cell_t[pos_decr]))
        pos_incr = pos_prefactor & (cell_diff < 0)
        p2 = torch.dot(mult_prefix[pos_incr], torch.tanh(self.cell_target[pos_incr]))
        neg_decr = ~pos_prefactor & (cell_diff >= 0)
        p3 = torch.dot(mult_prefix[neg_decr], torch.tanh(self.cell_target[neg_decr]))
        neg_incr = ~pos_prefactor & (cell_diff < 0)
        p4 = torch.dot(mult_prefix[neg_incr], torch.tanh(self.cell_t[neg_incr]))
        lbda_tilde = p1 + p2 + p3 + p4
        return self.model.activation(lbda_tilde)

"""
Neural network models for point processes.

@author: manifold
"""
import numpy as np
import torch
from torch import Tensor
from torch import nn
from typing import Tuple, List


class NeuralCTLSTM(nn.Module):
    """
    A continuous-time LSTM, defined according to Eisner & Mei's article
    https://arxiv.org/abs/1612.09328
    """

    def __init__(self, input_size: int, hidden_dim: int):
        super(NeuralCTLSTM, self).__init__()
        input_size += 1
        self.input_size = input_size
        self.hidden_dim = hidden_dim
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

    def init_hidden(self, batch_size: int = 1):
        """
        Initialize the hidden state, the cell state and cell state target.
        The first dimension is the batch size.

        Returns:
            (hidden, cell_state, cell_target)
        """
        return (torch.zeros(batch_size, self.hidden_dim),
                torch.zeros(batch_size, self.hidden_dim),
                torch.zeros(batch_size, self.hidden_dim))

    def forward(self, dt: Tensor, seq_types: Tensor, hidden_ti, cell_ti, cell_target):
        """
        Forward pass of the CT-LSTM network.

        Computes the network parameters for the next interval :math:`(t_i,t_{i+1}]` from the input states given by
        at time :math:`t_{i}`, right before the update.

        Args:
            dt: time until next event
                Shape: batch
            seq_types: one-hot encoded event sequence types
                Shape: batch * input_size
            hidden_ti: prev. hidden state, decayed since last event
            cell_ti: prev. cell state, decayed since last event
            cell_target: prev. cell state target

        Returns:
            output: result of the output gate
            hidden_ti: hidden state
            cell_ti: cell state
            c_t_actual: decayed cell state
            c_target_i: cell target
            decay_i: decay
        """
        v = torch.cat((seq_types, hidden_ti), dim=1)
        inpt = self.input_g(v)
        forget = self.forget_g(v)
        input_target = self.input_target(v)
        forget_target = self.forget_target(v)
        output = self.output_g(v)
        # Not-quite-c
        z_i = self.z_gate(v)
        # Compute the decay parameter
        decay_i = self.decay_layer(v)
        # Update the cell state to c(t_i+)
        cell_i = forget * cell_ti + inpt * z_i
        # Update the cell state target
        cell_target = forget_target * cell_target + input_target * z_i
        # Decay the cell state to its value before the known next event at t+dt
        c_t_actual = (
                cell_target + (cell_i - cell_target) *
                torch.exp(-decay_i * dt[:, None])
        )
        hidden_i = output * torch.tanh(cell_i)
        h_t_actual = output * torch.tanh(c_t_actual)  # h(ti)
        # Return our new states for the next pass to use
        return output, hidden_i, h_t_actual, cell_i, c_t_actual, cell_target, decay_i

    def compute_intensity(self, dt: Tensor, output: Tensor,
                          cell_ti: Tensor, c_target_i: Tensor, decay: Tensor) -> Tensor:
        """
        Compute the intensity function.

        Args:
            dt: time increments
                dt[i] is the time elapsed since event t_i
                if you want to compute at time s,
                :math:`t_i <= t <= t_{i+1}`, then `dt[i] = t - t_i`.
                Shape: seq_length * batch
            output: LSTM cell output
                Shape: seq_length * batch * hidden_dim
            cell_ti: previous cell state
            c_target_i: previous cell target
            decay: decay[i] is the degrowth param. on range [t_i, t_{i+1}]

        Shape:
            batch_size * max_seq_length * hidden_dim

        It is best to store the training history in variables for this.
        """
        # Get the values of c(t)
        c_t_after = (
                c_target_i + (cell_ti - c_target_i) *
                torch.exp(-decay * dt)
        )
        # Compute hidden state
        h_t = output * torch.tanh(c_t_after)
        return self.activation(h_t)

    def compute_loss(self, seq_times: Tensor, seq_types: Tensor, batch_sizes: Tensor, hiddens: List[Tensor],
                     cell_hist: List[Tensor], cell_target_hist: List[Tensor], outputs: List[Tensor],
                     decay_hist: List[Tensor], tmax: float) -> Tensor:
        """
        Compute the negative log-likelihood as a loss function.
        
        Args:
            seq_times: event occurrence timestamps
            seq_types: types of events in the sequence
            batch_sizes: batch sizes for each event sequence tensor, by length
            hiddens: hidden state history
            cell_hist: entire cell state history
            cell_target_hist: cell state target values history
            outputs: entire output history
            decay_hist: entire decay history
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """
        n_times = len(hiddens)
        dt_seq = seq_times[1:] - seq_times[:-1]
        # Get the intensity process
        intens_ev_times = [
            self.activation(hiddens[i])
            for i in range(n_times)
        ]
        # shape N * batch * K
        intens_ev_times = nn.utils.rnn.pad_sequence(
            intens_ev_times, batch_first=True, padding_value=1.0)
        intens_ev_times_filtered: Tensor = torch.sum(intens_ev_times * seq_types[:-1], dim=2)
        log_sum: Tensor = intens_ev_times_filtered.log().sum(dim=0)
        # The integral term is computed using a Monte Carlo method
        time_samples: Tensor = dt_seq*torch.rand_like(dt_seq)  # time increment samples for each interval
        intens_at_samples = [
            self.compute_intensity(time_samples[i, :batch_sizes[i]], outputs[i],
                                   cell_hist[i], cell_target_hist[i], decay_hist[i])
            for i in range(n_times)
        ]
        intens_at_samples = nn.utils.rnn.pad_sequence(
            intens_at_samples, batch_first=True, padding_value=0.0)  # shape N * batch * (K + 1)
        total_intens_samples: Tensor = intens_at_samples.sum(dim=2, keepdim=True)
        integral_estimates: Tensor = dt_seq * total_intens_samples
        second_term = integral_estimates.sum(dim=0)
        res = (- log_sum + second_term).mean()
        return res


class CTGenerator:
    """
    Sequence generator for the CT-LSTM model.
    """

    def __init__(self, model: NeuralCTLSTM):
        self.model = model
        with torch.no_grad():
            hidden, c_t, c_target = model.init_hidden()
            self.hidden_t = hidden
            self.cell_t = c_t
            self.cell_target = c_target
            self.hidden_hist = []
        self.sequence_ = []
        self.lbda_max_seq_ = []
        self.output = None
        self.cell_decay = None

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

    def update_hidden_state(self, t):
        """
        Update all cell states using forward pass on the model when an event occurs at t.
        """
        with torch.no_grad():
            output, hidden_i, cell_i, c_t_actual, c_target_i, decay_i = self.model.forward(
                t - self.sequence_[-1],
                self.hidden_t,
                self.cell_t,
                self.cell_target
            )
            self.output = output
            self.hidden_t = hidden_i  # New hidden state at t, start value on [t,\infty)
            self.cell_t = cell_i  # New cell state at t, start value on [t,\infty)
            self.cell_target = c_target_i  # New cell state target
            self.cell_decay = decay_i  # New decay parameter until next event
            self.hidden_hist.append({
                "hidden": self.hidden_t,
                "cell": self.cell_t,
                "cell_target": self.cell_target,
                "cell_decay": self.cell_decay,
                "output": self.output
            })

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

    def make_ctlstm_sequence_plot(self, n: int, tmax: float):
        """
        Make an intensity plot for the CTLSTM model

        Args:
            self: Neural Hawkes generator instance
            n: number of samples
            tmax: max time horizon
        """
        sequence = self.sequence_
        hidden_hist = self.hidden_hist
        tls = np.linspace(0, tmax, n)
        tls = np.sort(np.append(tls, sequence))
        interv_counter = 0
        y_vals = np.zeros_like(tls[:-1])
        for i in range(len(tls)):
            t = tls[i]
            if t > sequence[-1]:
                tls = tls[:i]
                y_vals = y_vals[:i]
                break
            while t > sequence[interv_counter]:
                interv_counter += 1
            c_t = hidden_hist[interv_counter]['cell']
            c_target = hidden_hist[interv_counter]['cell_target']
            output = hidden_hist[interv_counter]['output']
            decay = hidden_hist[interv_counter]['cell_decay']
            hidden_t = output * torch.tanh(
                c_target + (c_t - c_target) * torch.exp(-decay * (t - sequence[interv_counter]))
            )
            with torch.no_grad():
                y_vals[i] = self.model.activation(hidden_t).item()
        return tls, y_vals

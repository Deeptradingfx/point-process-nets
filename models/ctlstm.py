"""
Neural network models for point processes.

@author: manifold
"""
import numpy as np
import torch
import typing
from torch import nn
import pdb

device = torch.device('cpu')


class NeuralCTLSTM(nn.Module):
    """
    A continuous-time LSTM, defined according to Eisner & Mei's article
    https://arxiv.org/abs/1612.09328
    """

    def __init__(self, hidden_dim: int):
        super(NeuralCTLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_g = nn.Linear(hidden_dim, hidden_dim)
        self.forget_g = nn.Linear(hidden_dim, hidden_dim)
        self.output_g = nn.Linear(hidden_dim, hidden_dim)
        self.input_target = nn.Linear(hidden_dim, hidden_dim)
        self.forget_target = nn.Linear(hidden_dim, hidden_dim)
        # activation will be tanh
        self.z_gate = nn.Linear(hidden_dim, hidden_dim)
        # Cell decay factor, identical for all hidden dims
        self.decay_gate = nn.Linear(hidden_dim, 1)
        self.decay_act = nn.Softplus(beta=8.)
        # activation for the intensity
        self.w_alpha = nn.Linear(hidden_dim, 1, bias=False) # no bias in the model
        self.activation = nn.Softplus(beta=5.)

    def init_hidden(self):
        """
        Initialize the hidden state, the cell state and cell state target.
        The first dimension is the batch size.

        Returns:
            (hidden, cell_state, cell_target)
        """
        return (torch.zeros(1, self.hidden_dim, device=device),
                torch.zeros(1, self.hidden_dim, device=device),
                torch.zeros(1, self.hidden_dim, device=device))

    def forward(self, dt, hidden_ti, cell_i, c_target_i):
        """
        Forward pass for the CT-LSTM.

        The input states are given by at time :math:`t_{i-1}` before update.

        Computes the hidden states on interval :math:`(t_{i-1},t_i]`.

        Args:
            dt: time until next event
            hidden_ti: prev. hidden state
            cell_i: prev. cell state, decayed since last event
            c_target_i: prev. cell state target

        Returns:
            output: result of the output gate
            hidden_ti: hidden state
            cell_ti: cell state
            c_t_actual: decayed cell state
            c_target_i: cell target
            decay_i: decay parameter on the interval :math:`(t_{i-1},t_i]`
        TODO event type embedding
        """
        # TODO concatenate event embedding with h_t
        v = torch.cat((hidden_ti,))
        inpt = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        output = torch.sigmoid(self.output_g(v))
        input_target = torch.sigmoid(self.input_target(v))
        forget_target = torch.sigmoid(self.forget_target(v))
        # Not-quite-c
        z_i = torch.tanh(self.z_gate(v))
        # Compute the decay parameter
        decay_i = self.decay_act(self.decay_gate(v))
        # Update the cell state to c(t_i+)
        cell_i = forget * cell_i + inpt * z_i
        # Update the cell state target
        c_target_i = forget_target * c_target_i + input_target * z_i
        # Decay the cell state to its value before the known next event at t+dt
        c_t_actual = (
                c_target_i + (cell_i - c_target_i) *
                torch.exp(-decay_i*dt)
        )
        # h_t_actual = output * torch.tanh(c_t_actual)
        hidden_ti = output * torch.tanh(cell_i)
        # Return our new states for the next pass to use
        return output, hidden_ti, cell_i, c_t_actual, c_target_i, decay_i

    def compute_intensity(self, dt: torch.Tensor, output: torch.Tensor,
                          cell_ti, c_target_i, decay) -> torch.Tensor:
        """
        Compute the intensity function.

        Args:
            dt: time increments array
                dt[i] is the time elapsed since event t_i
                if you want to compute at time s,
                :math:`t_i <= t <= t_{i+1}`, then `dt[i] = t - t_i`.
            output: NN output o_i
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
                torch.exp(-decay * dt.unsqueeze(-1).expand(cell_ti.shape))
        )
        # Compute hidden state
        h_t = output * torch.tanh(c_t_after)
        pre_lambda = self.w_alpha(h_t)
        return self.activation(pre_lambda)

    def likelihood(self, event_times, seq_lengths, cell_hist, cell_target_hist,
                   output_hist, decay_hist, tmax) -> torch.Tensor:
        """
        Compute the negative log-likelihood as a loss function.
        
        Args:
            event_times: event occurrence timestamps
            seq_lengths: real sequence lengths
            cell_hist: entire cell state history
            cell_target_hist: cell state target values history
            output_hist: entire output history
            decay_hist: entire decay history
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """
        max_seq_length = event_times.size(0)
        increments = event_times[1:] - event_times[:-1]
        # Get the intensity process
        event_lambdas = self.compute_intensity(
            increments, output_hist,
            cell_hist, cell_target_hist, decay_hist)
        log_sum = event_lambdas.log().sum()
        # The integral term is computed using a Monte Carlo method
        n_samples = event_times.shape[0]
        all_samples: torch.Tensor = (
                tmax * torch.rand(*event_times.shape, n_samples)
        ) # uniform samples in [0, tmax]
        all_samples, _ = all_samples.sort(0)
        lam_samples_ = []
        for i in range(n_samples):
            samples = all_samples[:, i]
            dsamples = samples[1:] - samples[:-1]
            # Get the index of each elt of samples inside
            # the subdivision given by the event_times array
            # we have to substract 1 (indexes start at 1)
            # and the no. of trailing, padding 0s in the sequence
            indices = (
                torch.sum(samples[:-1,None] >= event_times[:-1], dim=1)
                - (max_seq_length-seq_lengths) - 1
            )
            # Get the samples of the intensity function
            try:
                lam_samples_.append(self.compute_intensity(
                    dsamples, output_hist[:,indices,:],
                    cell_hist[:,indices,:], cell_target_hist[:,indices,:],
                    decay_hist[:,indices,:]))
            except Exception as inst:
                pdb.set_trace()
                raise
        lam_samples_ = torch.stack(lam_samples_)
        integral = torch.sum(tmax * lam_samples_, dim=1)
        # Tensor of dim. batch_size
        # of the values of the likelihood
        res = -log_sum + integral
        return res.mean()

    def pred_loss(self, output, cell_hist, cell_target_hist):
        #
        pass


class CTGenerator:
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

    def generate_sequence(self, tmax: float) -> typing.List[torch.Tensor]:
        """
        Generate a sequence of events distributed according to the neural Hawkes model.

        WARNING: Check the model is in evaluation mode!
        """
        assert not self.model.training
        # Reinitialize sequence
        self.sequence_ = []
        s = 0.0
        self.sequence_.append(s)
        self.update_hidden_state(s)
        lbda_max = self.update_max_lambda()
        self.lbda_max_seq_.append(lbda_max)

        while s <= tmax:
            u1 = np.random.rand()
            dt: torch.Tensor = -1./lbda_max*np.log(u1)
            s += dt.item()  # Increment s
            print("Increment: {:}".format(dt.item()))
            if s > tmax:
                break
            t = self.sequence_[-1]
            # Compute the current intensity
            # Compute what the hidden state at s is
            cell_state_s = (
                    self.cell_target + (self.cell_t - self.cell_target)
                    * torch.exp(-self.cell_decay*(s-t))
            )
            hidden_state_s = self.output * torch.tanh(cell_state_s)
            # Apply activation function to hidden state
            intens = self.model.activation(self.model.w_alpha(hidden_state_s)).item()
            u2 = np.random.rand()  # random in [0,1]
            print("Intensity {:}".format(intens))
            print("\tlbda_max\t{:}".format(lbda_max))
            print("\tratio\t{:}".format(intens/lbda_max))
            if u2 <= intens/lbda_max:
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
        w_al = self.model.w_alpha.weight.data
        cell_diff = self.cell_t - self.cell_target
        mult_prefix = w_al*self.output
        pos_prefactor = mult_prefix > 0
        pos_decr = pos_prefactor & (cell_diff >= 0)
        p1 = torch.dot(mult_prefix[pos_decr], torch.tanh(self.cell_t[pos_decr]))
        pos_incr = pos_prefactor & (cell_diff < 0)
        p2 = torch.dot(mult_prefix[pos_incr], torch.tanh(self.cell_target[pos_incr]))
        neg_decr = ~pos_prefactor & (cell_diff >= 0)
        p3 = torch.dot(mult_prefix[neg_decr], torch.tanh(self.cell_target[neg_decr]))
        neg_incr = ~pos_prefactor & (cell_diff < 0)
        p4 = torch.dot(mult_prefix[neg_incr], torch.tanh(self.cell_t[neg_incr]))
        lbda_tilde = p1+p2+p3+p4
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
                y_vals[i] = self.model.activation(
                    self.model.w_alpha(hidden_t)).item()
        return tls, y_vals

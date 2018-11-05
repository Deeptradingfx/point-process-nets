"""
Neural network models for point processes.

@author: manifold
"""
import torch
from torch import nn
import pdb

device = torch.device('cpu')


class NeuralCTLSTM(nn.Module):
    """
    A continuous-time LSTM, defined according to Eisner & Mei's article
    https://arxiv.org/abs/1612.09328
    Batch size of all tensors must be the first dimension.
    """

    def __init__(self, hidden_dim: int):
        super(NeuralCTLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_g = nn.Linear(hidden_dim, hidden_dim)
        self.forget_g = nn.Linear(hidden_dim, hidden_dim)
        self.output_g = nn.Linear(hidden_dim, hidden_dim)
        self.ibar = nn.Linear(hidden_dim, hidden_dim)
        self.fbar = nn.Linear(hidden_dim, hidden_dim)
        # activation will be tanh
        self.z_gate = nn.Linear(hidden_dim, hidden_dim)
        # Cell decay factor
        self.decay = nn.Linear(hidden_dim, hidden_dim)
        # we can learn the parameters of this
        self.decay_act = nn.Softplus(beta=6.)
        self.activation = nn.Softplus(beta=6.)
        self.weight_a = torch.rand(self.hidden_dim, device=device)

    def init_hidden(self, batch_size=1):
        """
        Initialize the hidden state, the cell state and cell state target.
        The first dimension is the batch size.
        """
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def next_event(self, output, dt, decay):
        # c_t_after = self.c_func(dt, c_ti, cbar, decay)
        # h_t_after = output * torch.tanh(c_t_after)
        # lbdaMax = h_t_after
        raise NotImplementedError

    def forward(self, dt, hidden_i, cell_i, c_target_i):
        """
        Forward pass for the CT-LSTM.

        Args:
            dt: event inter-arrival times
            hidden_i: prev. hidden state
            cell_i: prev. cell state
            c_target_i: prev. cell state target

        Returns:
            output: result of the output gate
            hidden_i: hidden state
            cell_i: cell state
            c_target_i: cell target
            decay_i: decay parameter on the interval
        TODO event type embedding
        """
        # TODO concatenate event embedding with ht
        v = torch.cat((hidden_i,))
        input = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        output = torch.sigmoid(self.output_g(v))
        input_target = torch.sigmoid(self.ibar(v))
        forget_target = torch.sigmoid(self.fbar(v))
        # Not-quite-c
        z_i = torch.tanh(self.z_gate(v))
        # Compute the decay parameter
        decay_i = self.decay_act(self.decay(v))
        # Update the cell
        cell_i = forget * cell_i + input * z_i
        # Update the cell state target
        c_target_i = forget_target * c_target_i + input_target * z_i
        c_t_actual = (
            c_target_i + (cell_i - c_target_i) *
            torch.exp(-decay_i*dt)
        )
        # h_t_actual = output * torch.tanh(c_t_actual)
        hidden_i = output * torch.tanh(cell_i)
        # Return our new states for the next pass to use
        return output, hidden_i, cell_i, c_t_actual, c_target_i, decay_i

    def eval_intensity(self, dt: torch.Tensor, output: torch.Tensor,
                       c_ti, c_target_i, decay) -> torch.Tensor:
        """
        Compute the intensity function.

        Args:
            dt: time increments array
                dt[i] is the time elapsed since event t_i
                verify that if you want to compute at time s,
                t_i <= s <= t_{i+1}, then dt[i] = s - t_i
            output: NN output o_i
            c_ti: previous cell state
            c_target_i: previous cell target
            decay: decay[i] is the degrowth param. on range [t_i, t_{i+1}]

        Shape:
            batch_size * max_seq_length * hidden_dim

        It is best to store the training history in variables for this.
        """
        # Get the values of c(s)
        c_t_after = (
            c_target_i + (c_ti - c_target_i) *
            torch.exp(-decay * dt.unsqueeze(-1).expand(c_ti.shape))
        )
        # Compute hidden state
        h_t = output * torch.tanh(c_t_after)
        batch_size = h_t.size(0)
        hidden_size = self.weight_a.size(0)
        weight_a = (
            self.weight_a.expand(batch_size, hidden_size).unsqueeze(1)
        )
        pre_lambda = torch.bmm(weight_a, h_t.transpose(2, 1)).squeeze(1)
        return self.activation(pre_lambda)

    def likelihood(self, event_times, seq_lengths, cell_hist, cell_target_hist,
                   output_hist, decay_hist, T) -> torch.Tensor:
        """
        Compute the negative log-likelihood as a loss function.
        
        Args:
            event_times: event occurrence timestamps
            seq_lengths: real sequence lengths
            c_ti: entire cell state history
            output: entire output history
            decay: entire decay history
        
        Shape:
            []
        """
        max_seq_length = event_times.size(0)
        inter_times = event_times[1:] - event_times[:-1]
        # Get the intensity process
        event_lambdas = self.eval_intensity(
            inter_times, output_hist,
            cell_hist, cell_target_hist, decay_hist)
        log_sum = event_lambdas.log().sum()
        # The integral term is computed using a Monte Carlo method
        n_samples = event_times.shape[0]
        all_samples: torch.Tensor = (
            T*torch.rand(*event_times.shape, n_samples)
        )
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
                lam_samples_.append(self.eval_intensity(
                    dsamples, output_hist[:,indices,:],
                    cell_hist[:,indices,:], cell_target_hist[:,indices,:],
                    decay_hist[:,indices,:]))
            except Exception as inst:
                pdb.set_trace()
                raise
        lam_samples_ = torch.stack(lam_samples_)
        integral = torch.sum(inter_times*lam_samples_, dim=1)
        # Tensor of dim. batch_size
        # of the values of the likelihood
        res = -log_sum + integral
        # return the opposite of the mean
        return res.mean()

    def pred_loss(self, output, cell_hist, cell_target_hist):
        #
        pass

    def generate_seq(self, cell_hist, cell_target_hist, decay_hist):
        
        pass

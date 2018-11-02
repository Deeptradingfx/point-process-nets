"""
Point process models.

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
        self.decay_act = nn.Softplus()

        self.activation = nn.Softplus()
        self.weight_a = torch.rand(self.hidden_dim, device=device)

    def init_hidden(self, batch_size=1):
        """
        Initialize the hidden state, and the two hidden memory
        cells c and cbar.
        The first dimension is the batch size.
        """
        return (torch.randn(batch_size, self.hidden_dim, device=device),
                torch.randn(batch_size, self.hidden_dim, device=device),
                torch.randn(batch_size, self.hidden_dim, device=device))

    def next_event(self, output, dt, decay):
        # c_t_after = self.c_func(dt, c_ti, cbar, decay)
        # h_t_after = output * torch.tanh(c_t_after)
        # lbdaMax = h_t_after
        raise NotImplementedError

    def forward(self, inter_times, h_ti, c_ti, c_target_i):
        """
        inter_times: inter-arrival time for the next event in the sequence

        Args:
            inter_times:    interarrival times
            h_ti:           prev. hidden state
            c_ti:           prev. cell state
            c_target_i:     prev. cell state target

        Returns:
            output : result of the output gate
            h_ti   : hidden state
            c_ti   : cell state
            cbar   : cell target
            decay_t: decay parameter on the interval
        #TODO event type embedding
        """
        # TODO concatenate event embedding with ht
        v = torch.cat((h_ti,))
        input = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        output = torch.sigmoid(self.output_g(v))

        input_target = torch.sigmoid(self.ibar(v))
        forget_target = torch.sigmoid(self.fbar(v))

        # Not-quite-c
        zi = torch.tanh(self.z_gate(v))

        # Compute the decay parameter
        decay_t = self.decay_act(self.decay(v))

        # Now update the cell memory
        # Decay the cell memory
        # c_t_actual = self.c_func(inter_times, c_ti, c_target_i, decay_t)
        # Update the cell
        c_ti = forget * c_ti + input * zi
        # Update the cell state target
        c_target_i = forget_target * c_target_i + input_target * zi
        c_t_actual = (
            c_target_i + (c_ti - c_target_i) *
            torch.exp(-decay_t*inter_times.unsqueeze(1))
        )
        # h_t_actual = output * torch.tanh(c_t_actual)
        h_ti = output * torch.tanh(c_ti)

        # Store our new states for the next pass to use
        return output, h_ti, c_ti, c_t_actual, c_target_i, decay_t

    def eval_intensity(self, dt: torch.Tensor, output: torch.Tensor,
                       c_ti, c_target_i, decay) -> torch.Tensor:
        """
        Compute the intensity function
        Args:
            dt  time increments array
                dt[i] is the time elapsed since event t_i
                verify that if you want to compute at time t,
                t_i <= t <= t_{i+1}, then dt[i] = t - t_i
            output  NN output o_i
            c_ti    previous cell state
            c_target_i  previous cell target
            decay   decay[i] is the degrowth param. on range [t_i, t_{i+1}]

        It is best to store the training history in variables for this.
        """
        # Get the updated c(t)
        c_t_after = (
            c_target_i + (c_ti - c_target_i) *
            torch.exp(-decay * dt.unsqueeze(-1).expand(c_ti.shape))
        )

        h_t = output * torch.tanh(c_t_after)
        batch_size = h_t.size(0)
        try:
            hidden_size = self.weight_a.size(0)
            weight_a = (
                self.weight_a.expand(batch_size, hidden_size).unsqueeze(1)
            )
            pre_lambda = torch.bmm(weight_a, h_t.transpose(2, 1)).squeeze(1)
        except BaseException:
            print("Error occured in c_func")
            print(" dt shape %s" % str(dt.shape))
            print(" Weights shape %s" % str(self.weight_a.shape))
            print(" h_t shape %s" % str(h_t.shape))

            raise
        return self.activation(pre_lambda)

    def likelihood(self, event_times, cell_hist, cell_target_hist,
                   output_hist, decay_hist, T):
        """
        Compute the negative log-likelihood as a loss function
        #lengths: real sequence lengths
        c_ti :  entire cell state history
        output: entire output history
        decay:  entire decay history
        """
        inter_times: torch.Tensor = event_times[:, -1:] - event_times[:, 1:]
        # Get the intensity process
        event_intensities = self.eval_intensity(
            inter_times, output_hist,
            cell_hist, cell_target_hist, decay_hist)
        first_sum = event_intensities.log().sum(dim=1)
        # The integral term is computed using a Monte Carlo method
        M = 10  # Monte Carlo samples on each dimension
        samples: torch.Tensor = (
            inter_times.expand(M, *inter_times.shape) *
            torch.rand(M, *inter_times.shape)
        )

        # Get the samples of the intensity function
        lam_samples = torch.stack([self.eval_intensity(
            samples[i], output_hist, cell_hist,
            cell_target_hist, decay_hist) for i in range(M)]).mean(dim=0)
        integral = torch.sum(inter_times*lam_samples, dim=1)
        # Tensor of dim. batch_size
        # of the values of the likelihood
        res = first_sum - integral
        # return the opposite of the mean
        return - res.mean()

    def pred_loss(self, output, cell_hist, cell_target_hist):
        #
        pass

    def generate_seq(self, cell_hist, cell_target_hist):
        pass

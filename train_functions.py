import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from models.ctlstm import NeuralCTLSTM
from models.decayrnn import HawkesDecayRNN


def train_neural_ctlstm(nhlstm: NeuralCTLSTM, optimizer: torch.optim.Optimizer,
                        event_times, sequence_length, input_size, tmax: float):
    """Train the Neural Hawkes CTLSTM on input sequence

    Args:
        nhlstm: Hawkes CTLSTM model
        optimizer: optimizer instance
        event_times: input sequence of event timestamps
        sequence_length: real sequence length :math:`N`
        input_size: maximal input size
        tmax: max time horizon
    """
    # inter-arrival times
    dt = event_times[1:] - event_times[:-1]
    # event_times = split_into_batches[i]
    output_hist = []
    hidden_hist = []
    cell_hist = []
    cell_target_hist = []
    decay_hist = []
    # Initialize hidden and cell state at 0
    hidden, cell_state, cell_target = nhlstm.init_hidden()
    # Reset gradients; in PyTorch they accumulate
    nhlstm.zero_grad()
    # Loop over event times
    # First pass takes care of the interval [0,t1) before first event
    for j in range(input_size - 1):
        output, hidden, cell_state, _, cell_target, decay_t = nhlstm(
            dt[j], hidden, cell_state, cell_target)
        output_hist.append(output)
        hidden_hist.append(hidden)
        cell_hist.append(cell_state)
        cell_target_hist.append(cell_target)
        decay_hist.append(decay_t)
    output_hist = torch.stack(output_hist, dim=1)
    hidden_hist = torch.stack(hidden_hist, dim=1)
    cell_hist = torch.stack(cell_hist, dim=1)
    cell_target_hist = torch.stack(cell_target_hist, dim=1)
    decay_hist = torch.stack(decay_hist, dim=1)
    loss = nhlstm.likelihood(
        event_times, sequence_length, cell_hist, cell_target_hist,
        output_hist, decay_hist, tmax)
    # Compute the gradients
    loss.backward()
    # Update the model parameters
    optimizer.step()
    hist_dict = dict(
        output=output_hist,
        hidden=hidden_hist,
        cell_state=cell_hist,
        cell_target=cell_target_hist,
        decay_cell=decay_hist
    )
    return loss, hist_dict


def train_decayrnn(model: HawkesDecayRNN, optimizer: Optimizer, sequence: Tensor,
                   seq_types: Tensor, seq_lengths: Tensor, tmax: float, cuda: bool = False):
    """Train the HawkesDecayRNN model on the input data sequence

    Args:
        model: recurrent neural net model
        tmax:
        sequence: input event sequence
        seq_types: event types
        seq_lengths: sequence length
        optimizer:
        cuda:
    """
    max_seq_length = seq_lengths[0]
    sequence = sequence[:max_seq_length]
    seq_types = seq_types[:max_seq_length]
    # Embed the sequence into the event arrival intervals
    sequence = torch.cat((torch.zeros_like(sequence[:1]), sequence))
    dt_sequence = sequence[1:] - sequence[:-1]
    # Trim the sequence to its real length
    packed_times = nn.utils.rnn.pack_padded_sequence(dt_sequence, seq_lengths)
    # packed_types = nn.utils.rnn.pack_padded_sequence(seq_types, seq_lengths)
    # Reshape to a format the RNN can understand
    # N * batch
    max_batch_size = packed_times.batch_sizes[0]
    # Data records
    # hidd_decayed: 0
    # decay: 0
    hidd_decayed, decay = model.initialize_hidden(max_batch_size)
    hiddens = []
    decays = []
    for i in range(max_seq_length):
        # event t_i is happening
        batch_size = packed_times.batch_sizes[i]
        # hidden state just before this event
        hidd_decayed = hidd_decayed[:batch_size]
        # time until next event t_{i+1}
        dt_batch = dt_sequence[i, :batch_size]
        types_batch = seq_types[i, :batch_size]
        hidd, decay, hidd_decayed = model(dt_batch, types_batch, hidd_decayed)
        hiddens.append(hidd)
        decays.append(decay)
    train_data = {
        "hidden": hiddens,
        "decay": decays
    }
    loss: Tensor = model.compute_loss(sequence.unsqueeze(2), seq_types,
                                      packed_times.batch_sizes, hiddens, decays, tmax)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()  # zero the gradients
    return train_data, loss.item()

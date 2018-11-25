import torch
from torch import Tensor
from torch import nn
import tqdm
import numpy as np
import sys
from torch.optim import Optimizer
from models.ctlstm import NeuralCTLSTM
from models.decayrnn import HawkesDecayRNN
from typing import List


def train_neural_ctlstm(nhlstm: NeuralCTLSTM, optimizer: Optimizer,
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


def train_decayrnn(model: HawkesDecayRNN, optimizer: Optimizer, seq_times: Tensor, seq_types: Tensor,
                   seq_lengths: Tensor, tmax: float, batch_size: int, epochs: int,
                   use_jupyter: bool = False) -> List[float]:
    """
    Train the HawkesDecayRNN model.

    Args:
        seq_times: event sequence samples
        seq_types: event types of the event sequence
        seq_lengths: lengths of the sequence in the sample
        batch_size:
        model: recurrent neural net model
        epochs:
        optimizer:
        tmax:
        use_jupyter: use tqdm's Jupyter mode
    """
    model.train()  # ensure model is in training mode
    print("Batch size {}".format(batch_size))
    print("Number of epochs {}".format(epochs))
    train_size = seq_times.size(1)
    loss_hist = []
    for e in range(1, epochs + 1):
        # Epoch loop
        epoch_loss = []
        if use_jupyter:
            tr_loop_range = tqdm.tnrange(0, train_size, batch_size,
                                         file=sys.stdout, desc="Epoch %d" % e)
        else:
            tr_loop_range = tqdm.trange(0, train_size, batch_size,
                                        file=sys.stdout, desc="Epoch %d" % e)
        # Full pass through the dataset
        for i in tr_loop_range:
            optimizer.zero_grad()
            sub_seq_lengths = seq_lengths[i:(i + batch_size)]
            max_seq_length = sub_seq_lengths[0]
            sub_seq_times = seq_times[:max_seq_length+1, i:(i + batch_size)]
            sub_seq_types = seq_types[:max_seq_length+1, i:(i + batch_size)]
            # Inter-event time intervals
            dt_sequence = sub_seq_times[1:] - sub_seq_times[:-1]
            # print("max seq. lengths: {}".format(max_seq_length))
            # print("dt shape: {}".format(dt_sequence.shape))
            # Trim the sequence to its real length
            packed_times = nn.utils.rnn.pack_padded_sequence(dt_sequence, sub_seq_lengths)
            # packed_types = nn.utils.rnn.pack_padded_sequence(sub_seq_types, sub_seq_lengths)
            # Reshape to a format the RNN can understand
            # N * batch
            max_batch_size = packed_times.batch_sizes[0]
            # Data records
            # hidd_decayed: 0
            # decay: 0
            hidd_decayed, decay = model.initialize_hidden(max_batch_size)
            hiddens = []
            decays = []
            for j in range(max_seq_length):
                # event t_i is happening
                sub_batch_size = packed_times.batch_sizes[j]
                # hidden state just before this event
                hidd_decayed = hidd_decayed[:sub_batch_size]
                # time until next event t_{i+1}
                dt_batch = dt_sequence[j, :sub_batch_size]
                types_batch = sub_seq_types[j, :sub_batch_size]
                hidd, decay, hidd_decayed = model(dt_batch, types_batch, hidd_decayed)
                hiddens.append(hidd)
                decays.append(decay)
            train_data = {"hidden": hiddens,
                          "decay": decays}
            loss: Tensor = model.compute_loss(sub_seq_times.unsqueeze(2), sub_seq_types,
                                              packed_times.batch_sizes, hiddens, decays, tmax)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        epoch_loss_mean: float = np.mean(epoch_loss)
        loss_hist.append(epoch_loss_mean)  # append the final loss of each epoch
    return loss_hist

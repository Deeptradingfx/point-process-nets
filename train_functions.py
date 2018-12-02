"""
Training loops for the various models.
"""
import torch
from torch import Tensor
from torch import nn
import tqdm
import numpy as np
import sys
from torch.optim import Optimizer
from models.ctlstm import HawkesCTLSTM
from models import decayrnn
from models.decayrnn import HawkesDecayRNN
from utils.load_synth_data import one_hot_embedding
from typing import List, Dict, Tuple


def train_neural_ctlstm(model: HawkesCTLSTM, optimizer: Optimizer,
                        seq_times: Tensor, seq_types: Tensor,
                        seq_lengths: Tensor, tmax: float, batch_size: int,
                        n_epochs: int, use_jupyter: bool = False):
    """Train the Neural Hawkes CTLSTM on input sequence

    Args:
        model: Hawkes CTLSTM model
        optimizer: optimizer instance
        seq_times: input sequence of event timestamps
        seq_types: input sequence event types
        seq_lengths: real sequence lengths
        tmax: max time horizon
        batch_size: batch size
        n_epochs: number of epochs
        use_jupyter: use tqdm's Jupyter mode
    """
    model.train()  # ensure model is in training mode
    print("Batch size {}".format(batch_size))
    print("Number of epochs {}".format(n_epochs))
    # Reorder by decreasing order for PyTorch to understand
    seq_lengths, reorder_indices_ = seq_lengths.sort(descending=True)
    # Reorder by descending sequence length
    seq_times = seq_times[:, reorder_indices_]
    seq_types = seq_types[:, reorder_indices_]

    train_size = seq_times.size(1)
    loss_hist = []
    for e in range(1, n_epochs + 1):
        # Epoch loop
        epoch_loss = []
        if use_jupyter:
            tr_loop_range = tqdm.tnrange(0, train_size, batch_size,
                                         file=sys.stdout, desc="Epoch %d" % e)
        else:
            tr_loop_range = tqdm.trange(0, train_size, batch_size, ascii=True,
                                        file=sys.stdout, desc="Epoch %d" % e)
        # inter-arrival times
        for i in tr_loop_range:
            optimizer.zero_grad()
            batch_seq_lengths: Tensor = seq_lengths[i:(i + batch_size)]
            max_seq_length = batch_seq_lengths[0]
            batch_seq_times: Tensor = seq_times[:max_seq_length + 1, i:(i + batch_size)]
            batch_seq_types: Tensor = seq_types[:max_seq_length + 1, i:(i + batch_size)]
            # Inter-event time intervals
            batch_dt: Tensor = batch_seq_times[1:] - batch_seq_times[:-1]
            # print("max seq. lengths: {}".format(max_seq_length))
            # print("dt shape: {}".format(dt_sequence.shape))
            # Trim the sequence to its real length
            packed_times = nn.utils.rnn.pack_padded_sequence(batch_dt, batch_seq_lengths)
            # packed_types = nn.utils.rnn.pack_padded_sequence(sub_seq_types, sub_seq_lengths)
            # Reshape to a format the RNN can understand
            # N * batch
            max_batch_size = packed_times.batch_sizes[0]
            hidden_t, cell_state, cell_target = model.init_hidden(max_batch_size)
            cell_t = cell_state.clone()
            # event_times = split_into_batches[i]
            output_hist: List[Tensor] = []
            hidden_hist: List[Tensor] = []
            cell_hist = []
            cell_target_hist = []
            decay_hist = []
            # Initialize hidden and cell state at 0
            # Reset gradients; in PyTorch they accumulate
            # Loop over event times
            # First pass takes care of the interval [0,t1) before first event
            for j in range(max_seq_length):
                # event t_i is happening
                sub_batch_size = packed_times.batch_sizes[j]
                # hidden state just before this event
                hidden_t = hidden_t[:sub_batch_size]
                cell_t = cell_t[:sub_batch_size]
                cell_target = cell_target[:sub_batch_size]
                # time until next event t_{i+1}
                dt_batch = batch_dt[j, :sub_batch_size]
                types_batch = batch_seq_types[j, :sub_batch_size]

                output, hidden_i, hidden_t, cell_state, cell_t, cell_target, decay = model.forward(
                    dt_batch, types_batch, hidden_t, cell_t, cell_target)
                output_hist.append(output)
                hidden_hist.append(hidden_i)
                cell_hist.append(cell_state)
                cell_target_hist.append(cell_target)
                decay_hist.append(decay)
            loss = model.compute_loss(
                batch_seq_times.unsqueeze(2), batch_seq_types, packed_times.batch_sizes, hidden_hist, cell_hist,
                cell_target_hist, output_hist, decay_hist, tmax)
            # Compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()
            epoch_loss.append(loss.item())
            hist_dict = dict(
                output=output_hist,
                hidden=hidden_hist,
                cell_state=cell_hist,
                cell_target=cell_target_hist,
                decay_cell=decay_hist
            )
        epoch_loss_mean: float = np.mean(epoch_loss)
        loss_hist.append(epoch_loss_mean)  # append the final loss of each epoch
    return loss_hist


def train_decayrnn(model: HawkesDecayRNN, optimizer: Optimizer, seq_times: Tensor, seq_types: Tensor,
                   seq_lengths: Tensor, tmax: float, batch_size: int, n_epochs: int,
                   use_cuda: bool = False, use_jupyter: bool = False) -> Tuple[List[float], List[dict]]:
    """
    Train the HawkesDecayRNN model.

    Args:
        use_cuda:
        seq_times: event sequence samples
        seq_types: event types of the event sequence
        seq_lengths: lengths of the sequence in the sample
        batch_size:
        model: recurrent neural net model
        n_epochs:
        optimizer:
        tmax:
        use_jupyter: use tqdm's Jupyter mode
    """
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model.train()  # ensure model is in training mode
    print("Batch size {}".format(batch_size))
    print("Number of epochs {}".format(n_epochs))
    # Reorder by decreasing sequence length for pack_padded_sequence to understand
    seq_lengths, reorder_indices_ = seq_lengths.sort(descending=True)
    seq_times = seq_times[reorder_indices_]
    seq_types = seq_types[reorder_indices_]
    train_size = seq_times.size(0)
    print("Train size: {}".format(train_size))
    loss_hist = []
    train_hist = []
    for epoch in range(1, n_epochs + 1):
        # Epoch loop
        epoch_loss = []
        train_hist = []
        if use_jupyter:
            tr_loop_range = tqdm.tnrange(0, train_size, batch_size,
                                         file=sys.stdout, desc="Epoch %d" % epoch)
        else:
            tr_loop_range = tqdm.trange(0, train_size, batch_size, ascii=True,
                                        file=sys.stdout, desc="Epoch %d" % epoch)
        # Full pass through the dataset
        for i in tr_loop_range:
            optimizer.zero_grad()
            # Get the batch data
            batch_seq_lengths: Tensor = seq_lengths[i:(i + batch_size)]
            max_seq_length = batch_seq_lengths[0]
            batch_seq_times = seq_times[i:(i + batch_size), :max_seq_length+1]
            batch_seq_types = seq_types[i:(i + batch_size), :max_seq_length+1]
            # Inter-event time intervals
            batch_dt = batch_seq_times[:, 1:] - batch_seq_times[:, :-1]
            # print("max seq. lengths: {}".format(max_seq_length))
            # print("dt shape: {}".format(dt_sequence.shape))
            # Pack the sequences
            packed_dt = nn.utils.rnn.pack_padded_sequence(batch_dt, batch_seq_lengths, batch_first=True)
            packed_types = nn.utils.rnn.pack_padded_sequence(batch_seq_types, batch_seq_lengths, batch_first=True)
            max_pack_batch_size = packed_dt.batch_sizes[0]
            hidden0, decay = model.init_hidden(max_pack_batch_size, device)
            # Data records
            hiddens, decays, hiddens_ti = model(packed_dt, packed_types, hidden0)
            batch_onehot = one_hot_embedding(batch_seq_types, model.input_size)
            batch_onehot = batch_onehot[:, :, :model.process_dim]
            loss: Tensor = model.compute_loss(batch_seq_times, batch_onehot,
                                              packed_dt.batch_sizes, hiddens, hiddens_ti,
                                              decays, tmax)
            train_hist.append({"hidden": hiddens,
                               "decay": decays})
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        epoch_loss_mean: float = np.mean(epoch_loss)
        print('epoch {}: train loss {:.4f}'.format(epoch, epoch_loss_mean))
        loss_hist.append(epoch_loss_mean)  # append the final loss of each epoch
        model.trained_epochs += 1
    return loss_hist, train_hist


def plot_loss(epochs: int, loss_hist, title: str = None, log: bool = False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
    epochs_range = list(range(1, epochs + 1))
    ax.plot(epochs_range, loss_hist, color='red',
            linewidth=.7, marker='.', markersize=6)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    if log:
        ax.set_yscale('log')
    if title:
        ax.set_title = title
    return fig

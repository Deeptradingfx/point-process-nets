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
from models.ctlstm import HawkesLSTM
from models.decayrnn import HawkesDecayRNN
from utils.load_synth_data import one_hot_embedding
from typing import List, Dict, Tuple


def train_lstm(model: HawkesLSTM, optimizer: Optimizer,
               seq_times: Tensor, seq_types: Tensor,
               seq_lengths: Tensor, tmax: float, batch_size: int,
               n_epochs: int, use_cuda: bool = False, use_jupyter: bool = False):
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
        use_cuda: whether to use GPU computation or not
        use_jupyter: use tqdm's Jupyter mode
    """
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model.train()  # ensure model is in training mode
    print("Batch size {}".format(batch_size))
    print("Number of epochs {}".format(n_epochs))
    # Reorder by decreasing order for PyTorch to understand
    seq_lengths, reorder_indices_ = seq_lengths.sort(descending=True)
    # Reorder by descending sequence length
    seq_times = seq_times[reorder_indices_]
    seq_types = seq_types[reorder_indices_]

    # Size of the traing dataset
    train_size = seq_times.size(0)
    loss_hist = []
    train_hist = []
    for epoch in range(1, n_epochs + 1):
        # Epoch loop
        epoch_loss = []
        if use_jupyter:
            tr_loop_range = tqdm.tnrange(0, train_size, batch_size,
                                         file=sys.stdout, desc="Epoch %d" % epoch)
        else:
            tr_loop_range = tqdm.trange(0, train_size, batch_size, ascii=True,
                                        file=sys.stdout, desc="Epoch %d" % epoch)
        # import pdb; pdb.set_trace()
        # inter-arrival times
        for i in tr_loop_range:
            optimizer.zero_grad()
            # Get the batch data
            batch_seq_lengths: Tensor = seq_lengths[i:(i + batch_size)]
            max_seq_length = batch_seq_lengths[0]
            batch_seq_times = seq_times[i:(i + batch_size), :max_seq_length + 1]
            batch_seq_types = seq_types[i:(i + batch_size), :max_seq_length + 1]
            # Inter-event time intervals
            batch_dt = batch_seq_times[:, 1:] - batch_seq_times[:, :-1]
            # print("max seq. lengths: {}".format(max_seq_length))
            # print("dt shape: {}".format(dt_sequence.shape))
            # Pack the sequences
            packed_dt = nn.utils.rnn.pack_padded_sequence(batch_dt, batch_seq_lengths, batch_first=True)
            packed_types = nn.utils.rnn.pack_padded_sequence(batch_seq_types, batch_seq_lengths, batch_first=True)
            max_pack_batch_size = packed_dt.batch_sizes[0]
            h0, c0, c_target0 = model.init_hidden(max_pack_batch_size, device)
            # Data records
            hiddens_ti, outputs, cells, cell_targets, decays = model(
                packed_dt, packed_types, h0, c0, c_target0)
            batch_onehot = one_hot_embedding(batch_seq_types, model.input_size)
            batch_onehot = batch_onehot[:, :, :model.process_dim]

            loss: Tensor = model.compute_loss(batch_seq_times, batch_onehot,
                                              packed_dt.batch_sizes, hiddens_ti,
                                              cells, cell_targets, outputs, decays, tmax)

            # Compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()
            epoch_loss.append(loss.item())
            train_hist.append(dict(
                output=outputs,
                cell_state=cells,
                cell_target=cell_targets,
                decay_cell=decays
            ))
        epoch_loss_mean: float = np.mean(epoch_loss)
        print('epoch {}: train loss {:.4f}'.format(epoch, epoch_loss_mean))
        loss_hist.append(epoch_loss_mean)  # append the final loss of each epoch
        model.trained_epochs += 1
    return loss_hist, train_hist


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
            h0 = model.init_hidden(max_pack_batch_size, device)
            # Data records
            hiddens, decays, hiddens_ti = model(packed_dt, packed_types, h0)
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

import torch


def train_neural_ctlstm(nhlstm, optimizer, event_times,
                        sequence_length, input_size, tmax):
    """Train the Neural Hawkes CTLSTM on input sequence

    Args:
        event_times: input sequence of event timestamps
        sequence_length: real sequence length :math:`N`
        input_size: maximal input size
    """
    # inter-arrival times
    dt = event_times[1:] - event_times[:-1]
    # event_times = split_into_batches[i]
    # batch_size = inter_t.size(0)
    output_hist = []
    hidden_hist = []
    cell_hist = []
    cell_target_hist = []
    decay_hist = []
    # Initialize hidden and cell state at 0
    hidden, cti, cbar = nhlstm.init_hidden()
    # Reset gradients; in PyTorch they accumulate
    nhlstm.zero_grad()
    # Loop over event times
    # First pass takes care of the interval [0,t1) before first event
    for j in range(input_size - 1):
        output, hidden, cti, _, cbar, decay_t = nhlstm(
            dt[j], hidden, cti, cbar)
        output_hist.append(output)
        hidden_hist.append(hidden)
        cell_hist.append(cti)
        cell_target_hist.append(cbar)
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
        decay=decay_hist
    )
    return loss, hist_dict

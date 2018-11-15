import torch
from models.ctlstm import NeuralCTLSTM


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

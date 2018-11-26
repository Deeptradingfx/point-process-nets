import torch
from torch import nn


def process_loaded_sequences(loaded_hawkes_data: dict):
    """
    Preprocess synthetic Hawkes data.

    Args:
        loaded_hawkes_data:

    Returns:

    """
    # Tensor of sequence lengths
    seq_lengths = torch.Tensor(loaded_hawkes_data['lengths']).int()
    # Reorder by decreasing order for PyTorch to understand
    seq_lengths, reorder_indices_ = seq_lengths.sort(descending=True)

    event_times_list = loaded_hawkes_data['timestamps']
    event_types_list = loaded_hawkes_data['types']
    event_times_list = [torch.from_numpy(e) for e in event_times_list]
    event_types_list = [torch.from_numpy(e) for e in event_types_list]

    # Build a types data tensor by padding
    process_dim = max(e.max() for e in event_types_list)

    # Build a data tensor by padding
    times_tensor = nn.utils.rnn.pad_sequence(event_times_list).float()
    times_tensor = torch.cat((torch.zeros_like(times_tensor[:1, :]), times_tensor))
    # Reorder by descending sequence length
    times_tensor = times_tensor[:, reorder_indices_]

    types_tensor = nn.utils.rnn.pad_sequence(event_types_list, padding_value=process_dim+1)
    # K is the Beginning-of-sequence event now; shift all other event types
    types_tensor = torch.cat(
        ((process_dim + 1) * torch.ones_like(types_tensor[:1, :]), types_tensor))
    # Reorder by descending sequence length
    types_tensor = types_tensor[:, reorder_indices_]
    return times_tensor, types_tensor, seq_lengths


def one_hot_embedding(labels, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form.

    Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.

    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

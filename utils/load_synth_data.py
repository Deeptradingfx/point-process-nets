import torch
from torch import Tensor, nn
from typing import Tuple


def process_loaded_sequences(loaded_hawkes_data: dict, process_dim: int, tmax: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Preprocess synthetic Hawkes data by padding the sequences.

    Args:
        loaded_hawkes_data:
        process_dim:
        tmax:

    Returns:
        sequence event times, event types and overall lengths (dim0: batch size)
    """
    # Tensor of sequence lengths (with additional BOS event)
    seq_lengths = torch.Tensor(loaded_hawkes_data['lengths']).int()

    event_times_list = loaded_hawkes_data['timestamps']
    event_types_list = loaded_hawkes_data['types']
    event_times_list = [torch.from_numpy(e) for e in event_times_list]
    event_types_list = [torch.from_numpy(e) for e in event_types_list]

    # Build a data tensor by padding
    seq_times = nn.utils.rnn.pad_sequence(event_times_list, batch_first=True, padding_value=tmax).float()
    seq_times = torch.cat((torch.zeros_like(seq_times[:, :1]), seq_times), dim=1)

    seq_types = nn.utils.rnn.pad_sequence(event_types_list, batch_first=True, padding_value=process_dim)
    # 0 is the Beginning-of-sequence event now; shift all other event types by 1
    seq_types = torch.cat(
        (process_dim*torch.ones_like(seq_types[:, :1]), seq_types), dim=1)
    return seq_times, seq_types, seq_lengths


def one_hot_embedding(labels: Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.

    Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.

    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]

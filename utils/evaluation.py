import numpy as np

from models.base import SeqGenerator


def generate_multiple_sequences(generator: SeqGenerator, tmax: float, n_gen_seq: int = 100):
    """

    Args:
        generator:
        tmax: end time for the simulations
        n_gen_seq: number of samples to take
    """
    # Build a statistic for the no. of events
    gen_seq_lengths = []
    gen_seq_types_lengths = []
    for i in range(n_gen_seq):
        generator.generate_sequence(tmax, record_intensity=False)
        gen_seq_times = generator.event_times
        gen_seq_types = np.array(generator.event_types)
        gen_seq_lengths.append(len(gen_seq_times))
        gen_seq_types_lengths.append([
            (gen_seq_types == i).sum() for i in range(generator.model.input_size)
        ])
    gen_seq_lengths = np.array(gen_seq_lengths)
    gen_seq_types_lengths = np.array(gen_seq_types_lengths)

    print("Mean generated sequence length: {}".format(gen_seq_lengths.mean()))
    print("Generated sequence length std. dev: {}".format(gen_seq_lengths.std()))
    return gen_seq_lengths, gen_seq_types_lengths

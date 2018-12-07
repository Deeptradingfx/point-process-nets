import numpy as np
from torch import nn


class SeqGenerator:
    def __init__(self, model: nn.Module, record_intensity: bool = False):
        self.model = model
        self.process_dim = model.input_size - 1  # process dimension
        print("Process model dim:\t{}\tHidden units:\t{}".format(self.process_dim, model.hidden_size))
        self.event_times = []
        self.event_types = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self._plot_times = []
        self.event_intens = []
        self.record_intensity: bool = record_intensity

    def _restart_sequence(self):
        self.event_times = []
        self.event_types = []
        self.event_intens = []
        self.decay_hist = []
        self.hidden_hist = []
        self.intens_hist = []
        self._plot_times = []

    def generate_sequence(self, tmax: float, record_intensity: bool):
        raise NotImplementedError

    def plot_events_and_intensity(self, model_name: str = None, debug=False):
        import matplotlib.pyplot as plt
        gen_seq_times = self.event_times
        gen_seq_types = self.event_types
        sequence_length = len(gen_seq_times)
        print("no. of events: {}".format(sequence_length))
        evt_times = np.array(gen_seq_times)
        evt_types = np.array(gen_seq_types)
        fig, ax = plt.subplots(1, 1, sharex='all', dpi=100,
                               figsize=(9, 4))
        ax: plt.Axes
        inpt_size = self.process_dim
        ax.set_xlabel('Time $t$ (s)')
        intens_hist = np.stack(self.intens_hist)[:, 0]
        labels = ["type {}".format(i) for i in range(self.process_dim)]
        for y, lab in zip(intens_hist.T, labels):
            ax.plot(self._plot_times, y, linewidth=.7, label=lab)
        ax.set_ylabel(r"Intensities $\lambda^i_t$")
        title = "Event arrival times and intensities for generated sequence"
        if model_name is None:
            model_name = self.model.__class__.__name__
        title += " ({})".format(model_name)
        ax.set_title(title)
        ylims = ax.get_ylim()
        ts_y = np.stack(self.event_intens)[:, 0]
        for k in range(inpt_size):
            mask = evt_types == k
            print(k, end=': ')
            if k == self.process_dim:
                print("starter type")
                # label = "start event".format(k)
                y = self.intens_hist[0].sum(axis=1)
            else:
                print("type {}".format(k))
                y = ts_y[mask, k]
                # label = "type {} event".format(k)
            ax.scatter(evt_times[mask], y, s=9, zorder=5,
                       alpha=0.8)
            ax.vlines(evt_times[mask], ylims[0], ylims[1], linewidth=0.3, linestyles='-', alpha=0.8)

        # Useful for debugging the sampling for the intensity curve.
        if debug:
            for s in self._plot_times:
                ax.vlines(s, ylims[0], ylims[1], linewidth=0.3, linestyles='--', alpha=0.6, colors='red')

        ax.set_ylim(*ylims)
        ax.legend()
        fig.tight_layout()
        return fig

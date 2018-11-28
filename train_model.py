"""
Module for model training.
"""
import argparse
import datetime
import glob
import os
import pickle

import torch
from torch import optim

from utils.load_synth_data import process_loaded_sequences, one_hot_embedding
from models.decayrnn import HawkesDecayRNN
from train_functions import train_decayrnn

SEED = 52
torch.manual_seed(SEED)
DEFAULT_BATCH_SIZE = 24
DEFAULT_HIDDEN_SIZE = 3
DEFAULT_LEARN_RATE = 0.015

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help='number of epochs.')
    parser.add_argument('-d', '--dim', type=int, required=True,
                        help='number of event types.')
    parser.add_argument('-b', '--batch', type=int,
                        dest='batch_size', default=DEFAULT_BATCH_SIZE,
                        help='batch size. (default: {})'.format(DEFAULT_BATCH_SIZE))
    parser.add_argument('--learning-rate', default=DEFAULT_LEARN_RATE, type=float,
                        help="set the optimizer learning rate. (default {})".format(DEFAULT_LEARN_RATE))
    parser.add_argument('--hidden', type=int,
                        dest='hidden_size', default=DEFAULT_HIDDEN_SIZE,
                        help='number of hidden units. (default: {})'.format(DEFAULT_HIDDEN_SIZE))
    parser.add_argument('--log-dir', type=str,
                        dest='log_dir', default='logs',
                        help="training logs target directory.")
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help="do not save the model state dict and loss history.")
    parser.add_argument('--cuda', dest='use_cuda', action='store_true',
                        help="whether or not to use GPU.")

    args = parser.parse_args()
    USE_CUDA = args.use_cuda

    SYNTH_DATA_FILES = glob.glob("data/simulated/*.pkl")
    print("Available files:")
    for i, s in enumerate(SYNTH_DATA_FILES):
        print("{:<4}{:<10}".format(i, s))

    process_dim = args.dim
    print("Loading {}-dimensional process.".format(process_dim), end=' ')
    chosen_file_index = int(input("Which file ? Index: "))
    chosen_file = SYNTH_DATA_FILES[chosen_file_index]
    with open(chosen_file, 'rb') as f:
        loaded_hawkes_data = pickle.load(f)

    mu = loaded_hawkes_data['mu']
    decay = loaded_hawkes_data['decay']
    tmax = loaded_hawkes_data['tmax']

    print("Hawkes process parameters:")
    for label, val in [("mu", mu), ("decay", decay), ("tmax", tmax)]:
        print("{:<20}{}".format(label, val))

    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print("Training on device {}".format(device))
    times_tensor, seq_types, seq_lengths = process_loaded_sequences(loaded_hawkes_data)
    onehot_types = one_hot_embedding(seq_types, process_dim + 1)
    times_tensor = times_tensor.to(device)
    seq_types = seq_types.to(device)
    seq_lengths = seq_lengths.to(device)
    onehot_types = onehot_types.to(device)

    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    print("Hidden size: {}".format(hidden_size))
    model = HawkesDecayRNN(process_dim, hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), learning_rate)

    total_sample_size = times_tensor.size(1)
    train_size = 1400
    print("Train sample size: {:}/{:}".format(train_size, total_sample_size))

    # Define training data
    train_times_tensor = times_tensor[:, :train_size]
    train_onehot_types = onehot_types[:, :train_size]
    train_seq_lengths = seq_lengths[:train_size]

    # Training parameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    loss_hist = train_decayrnn(model, optimizer, train_times_tensor, train_onehot_types, train_seq_lengths,
                               tmax, BATCH_SIZE, EPOCHS, use_cuda=USE_CUDA, use_jupyter=False)

    if args.save:
        # Model file dump
        SAVED_MODELS_PATH = os.path.abspath('saved_models')
        os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
        # print("Saved models directory: {}".format(SAVED_MODELS_PATH))

        date_format = "%Y%m%d-%H%M%S"
        now_timestamp = datetime.datetime.now().strftime(date_format)
        extra_tag = "{}d".format(process_dim)
        filename_base = "{}-{}-{}".format(model.__class__.__name__, extra_tag, now_timestamp)
        filename_model_save = filename_base + ".pth"
        filepath = os.path.join(SAVED_MODELS_PATH, filename_model_save)
        print("Saving model state dict to {}".format(filepath))
        torch.save(model.state_dict(), filepath)

        # Save train history to logs
        LOGS_PATH = os.path.abspath(args.log_dir)
        os.makedirs(LOGS_PATH, exist_ok=True)
        # print("Logs directory: {}".format(LOGS_PATH))
        filepath_loss_hist = os.path.join(LOGS_PATH, "log_" + filename_base + ".pkl")
        print("Saving traning loss log to {}".format(filepath_loss_hist))
        with open(filepath_loss_hist, 'wb') as f:
            pickle.dump(loss_hist, f)

        try:
            from train_functions import plot_loss
            fig = plot_loss(EPOCHS, loss_hist)
            filename_loss_plot = "loss_plot_" + filename_base + ".png"
            loss_plot_filepath = os.path.join(LOGS_PATH, filename_loss_plot)
            print("Saving loss plot to {}".format(loss_plot_filepath))
            fig.savefig(loss_plot_filepath)
        except ImportError:
            print("Error importing matplotlib.")

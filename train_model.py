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

from utils.load_synth_data import process_loaded_sequences
from models import HawkesDecayRNN, HawkesLSTM
from train_functions import train_decayrnn, train_lstm

DEFAULT_BATCH_SIZE = 24
DEFAULT_HIDDEN_SIZE = 12
DEFAULT_LEARN_RATE = 0.02

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help='number of epochs.')
    parser.add_argument('-d', '--dim', type=int, required=True,
                        help='number of event types.')
    parser.add_argument('-b', '--batch', type=int,
                        dest='batch_size', default=DEFAULT_BATCH_SIZE,
                        help='batch size. (default: {})'.format(DEFAULT_BATCH_SIZE))
    parser.add_argument('--lr', default=DEFAULT_LEARN_RATE, type=float,
                        help="set the optimizer learning rate. (default {})".format(DEFAULT_LEARN_RATE))
    parser.add_argument('--hidden', type=int,
                        dest='hidden_size', default=DEFAULT_HIDDEN_SIZE,
                        help='number of hidden units. (default: {})'.format(DEFAULT_HIDDEN_SIZE))
    parser.add_argument('--train-size', type=int,
                        help='override the size of the training dataset.')
    parser.add_argument('--log-dir', type=str,
                        dest='log_dir', default='logs',
                        help="training logs target directory.")
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help="do not save the model state dict and loss history.")
    parser.add_argument('--cuda', dest='use_cuda', action='store_true',
                        help="whether or not to use GPU.")
    parser.add_argument('-m', '--model', required=True,
                        type=str, choices=['rnn', 'lstm'],
                        help='choose which model to train.')

    args = parser.parse_args()
    USE_CUDA = args.use_cuda

    SYNTH_DATA_FILES = glob.glob("data/simulated/*.pkl")
    print("Available files:")
    for i, s in enumerate(SYNTH_DATA_FILES):
        print("{:<8}{:<8}".format(i, s))

    process_dim = args.dim
    print("Loading {}-dimensional process.".format(process_dim), end=' ')
    chosen_file_index = int(input("Which file ? Index: "))
    chosen_file = SYNTH_DATA_FILES[chosen_file_index]
    with open(chosen_file, 'rb') as f:
        loaded_hawkes_data = pickle.load(f)

    mu = loaded_hawkes_data['mu']
    alpha = loaded_hawkes_data['alpha']
    decay = loaded_hawkes_data['decay']
    tmax = loaded_hawkes_data['tmax']

    print("Hawkes process parameters:")
    for label, val in [("mu", mu), ("alpha", alpha), ("decay", decay), ("tmax", tmax)]:
        print("{:<20}{}".format(label, val))

    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print("Training on device {}".format(device))
    seq_times, seq_types, seq_lengths = process_loaded_sequences(loaded_hawkes_data, process_dim, tmax)
    seq_times = seq_times.to(device)
    seq_types = seq_types.to(device)
    seq_lengths = seq_lengths.to(device)

    hidden_size = args.hidden_size
    learning_rate = args.lr
    MODEL_TOKEN = args.model

    model = None
    if MODEL_TOKEN == 'rnn':
        model = HawkesDecayRNN(process_dim, hidden_size).to(device)
    elif MODEL_TOKEN == 'lstm':
        model = HawkesLSTM(process_dim, hidden_size).to(device)
    else:
        exit()
    MODEL_NAME = model.__class__.__name__
    print("Chose model {}".format(MODEL_NAME))
    print("Hidden size: {}".format(hidden_size))
    optimizer = optim.Adam(model.parameters(), learning_rate)

    total_sample_size = seq_times.size(0)
    if args.train_size:
        train_size = args.train_size
    else:
        print("Total sample size: {}".format(total_sample_size))
        train_size = int(input("Training data size: "))
    print("Train sample size: {:}/{:}".format(train_size, total_sample_size))

    # Define training data
    train_times_tensor = seq_times[:train_size]
    train_seq_types = seq_types[:train_size]
    train_seq_lengths = seq_lengths[:train_size]

    # Training parameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    if MODEL_TOKEN == 'rnn':
        loss_hist = train_decayrnn(
            model, optimizer, train_times_tensor, train_seq_types, train_seq_lengths,
            tmax, BATCH_SIZE, EPOCHS, use_cuda=USE_CUDA, use_jupyter=False)
    elif MODEL_TOKEN == 'lstm':
        loss_hist = train_lstm(
            model, optimizer, train_times_tensor, train_seq_types, train_seq_lengths,
            tmax, BATCH_SIZE, EPOCHS, use_cuda=USE_CUDA, use_jupyter=False)
    else:
        exit()

    if args.save:
        # Model file dump
        SAVED_MODELS_PATH = os.path.abspath('saved_models')
        os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
        # print("Saved models directory: {}".format(SAVED_MODELS_PATH))

        date_format = "%Y%m%d-%H%M%S"
        now_timestamp = datetime.datetime.now().strftime(date_format)
        extra_tag = "{}d".format(process_dim)
        filename_base = "{}-{}_hidden{}-{}".format(
            MODEL_NAME, extra_tag,
            hidden_size, now_timestamp)
        from utils.save_model import save_model
        save_model(model, chosen_file, extra_tag,
                   hidden_size, now_timestamp, MODEL_NAME)

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

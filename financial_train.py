import os
import datetime
import torch
from torch import nn, optim
import numpy as np
from models import HawkesDecayRNN, HawkesLSTM
from train_functions import train_decayrnn, train_lstm, plot_loss
from utils.save_model import save_model
import pandas as pd
from argparse import ArgumentParser

pd.options.mode.chained_assignment = None

DEFAULT_LR = 1e-3
DEFAULT_BATCH = 32

parser = ArgumentParser(description="Train the model on financial data.")
parser.add_argument('--data', type=str, required=True, nargs='+',
                    help='Location to find the financial data file.')
parser.add_argument('-b', '--batch', type=int, dest='batch_size', default=DEFAULT_BATCH,
                    help='Batch size. (default {})'.format(DEFAULT_BATCH))
parser.add_argument('-m', '--model', required=True, choices={'rnn', 'lstm'},
                    help="Model to train.")
parser.add_argument('-e', '--epochs', type=int, required=True,
                    help='Number of epochs.')
parser.add_argument('--lr', type=float,
                    default=DEFAULT_LR, dest='learning_rate',
                    help="Learning rate. (default {})".format(DEFAULT_LR))
parser.add_argument('--cuda', action='store_true',
                    help="Whether or not to use GPU acceleration.")

args = parser.parse_args()
data_files = args.data  # data path
data_files.sort()

# Data processing
split_times_list = []
split_types_list = []
seq_lengths = []
num_of_splits = int(input("Number of splits: "))

process_dim = 2
hidden_size = 128

for file in data_files:
    df = pd.read_csv(file)
    day_stamp = df.Date.values[0]
    print("Day", day_stamp)
    df['MidPrice'] = 0.5*(df.AskPriceAfter + df.BidPriceAfter)
    df['MidPriceVar'] = df.MidPrice.diff()
    df_filtered: pd.DataFrame = df[df.MidPriceVar != 0]
    df_filtered['MidPriceChangeType'] = (df.MidPriceVar > 0).astype(int)
    # Arrange our data
    evt_times = df_filtered.Time.values
    evt_types = df_filtered.MidPriceChangeType.values

    sttl_ = np.array_split(evt_times, num_of_splits)
    stpl_ = np.array_split(evt_types, num_of_splits)
    seq_lengths += [len(e) for e in sttl_]
    split_times_list += [torch.from_numpy(e) for e in sttl_]
    split_types_list += [torch.from_numpy(e) for e in stpl_]
seq_lengths = torch.LongTensor(seq_lengths) - 2
seq_times = nn.utils.rnn.pad_sequence(split_times_list, batch_first=True).to(torch.float32)
seq_types = nn.utils.rnn.pad_sequence(split_types_list, batch_first=True,
                                      padding_value=process_dim)
print(seq_times.shape)


if args.cuda:
    seq_lengths = seq_lengths.cuda()
    seq_times = seq_times.cuda()
    seq_types = seq_types.cuda()

# Define model

if args.model == 'rnn':
    model = HawkesDecayRNN(process_dim, hidden_size)
elif args.model == 'lstm':
    model = HawkesLSTM(process_dim, hidden_size)
else:
    exit()
MODEL_NAME = model.__class__.__name__
print('Training model {}'.format(MODEL_NAME))
if args.cuda:
    model = model.cuda()
num_of_parameters = sum(e.numel() for e in model.parameters())
print("no. of model parameters:", num_of_parameters)

# Train model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
if args.model == 'rnn':
    loss = train_decayrnn(model, optimizer, seq_times, seq_types, seq_lengths, -1,  # tmax actually doesn't matter
                          BATCH_SIZE, EPOCHS, use_cuda=args.cuda)
elif args.model == 'lstm':
    loss = train_lstm(model, optimizer, seq_times, seq_types, seq_lengths, -1,  # tmax actually doesn't matter
                      BATCH_SIZE, EPOCHS, use_cuda=args.cuda)

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

save_model(model, data_files, extra_tag,
           hidden_size, now_timestamp, MODEL_NAME)

# Plot loss and save
filename_loss_plot = "logs/loss_plot_" + filename_base + ".png"
fig = plot_loss(EPOCHS, loss)
fig.savefig(filename_loss_plot)


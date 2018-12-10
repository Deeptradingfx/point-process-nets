import torch
from torch import nn, optim
import numpy as np
from models import HawkesDecayRNN
from train_functions import train_decayrnn
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser(description="Train the model on financial data.")
parser.add_argument('--data', type=str, required=True,
                    help='Location to find the financial data file.')
parser.add_argument('--cuda', action='store_true',
                    help="Whether or not to use GPU acceleration.")

args = parser.parse_args()
DATA_FILE = args.data  # data path

df = pd.read_csv(DATA_FILE)
df = df[df.OrderType != 0]  # remove cancel orders
day_stamp = df.Date.values[0]
print("Day:", day_stamp)

print(df)

# Arrange our data
evt_times = df.Time.values
evt_types = (df.OrderType.values + 1)//2  # 0 is ask, 1 is bid

num_of_splits = 1000
split_times_list = np.array_split(evt_times, num_of_splits)
split_types_list = np.array_split(evt_types, num_of_splits)
seq_lengths = [len(e) for e in split_times_list]
split_times_list = [torch.from_numpy(e) for e in split_times_list]
split_types_list = [torch.from_numpy(e) for e in split_types_list]
seq_lengths = torch.LongTensor(seq_lengths) - 1
seq_times = nn.utils.rnn.pad_sequence(split_times_list, batch_first=True).to(torch.float32)
seq_types = nn.utils.rnn.pad_sequence(split_types_list, batch_first=True)

# Define model
PROCESS_DIM = 2
HIDDEN_SIZE = 128

EPOCHS = 4
BATCH_SIZE = 32

model = HawkesDecayRNN(PROCESS_DIM, HIDDEN_SIZE)
num_of_paramters = sum(e.numel() for e in model.parameters())
print("no. of model parameters:", num_of_paramters)
optimizer = optim.Adam(model.parameters(), lr=5e-3)

loss = train_decayrnn(model, optimizer, seq_times, seq_types, seq_lengths, -1,  # tmax actually doesn't matter
                      BATCH_SIZE, EPOCHS, use_cuda=args.cuda)

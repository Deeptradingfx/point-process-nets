import torch
from torch import nn, optim
import os, sys, glob
import argparse
from load_synth_data import process_loaded_sequences, one_hot_embedding
from models.decayrnn import HawkesDecayRNN
from train_functions import train_decayrnn


SEED = 52
torch.manual_seed(SEED)

parser = argparse.ArgumentParser(description="Train the model.")
parser.add_argument('--epochs', type=int, required=True,
                    help='Number of epochs.')
parser.add_argument('--batch', type=int,
                    dest='batch_size', default=32,
                    help='Batch size.')

args = parser.parse_args()

SYNTH_DATA_FILES = glob.glob("data/simulated/*.pkl")
print("Available files:")
for i, s in enumerate(SYNTH_DATA_FILES):
    print("{:<4}{:<3}".format(i, s))

process_dim = 1
print("Loading {}-dimensional process".format(process_dim))
chosen_file_index = int(input("Index of file: "))
chosen_file = SYNTH_DATA_FILES[chosen_file_index]
with open(chosen_file, 'rb') as f:
    import pickle
    loaded_hawkes_data = pickle.load(f)

mu = loaded_hawkes_data['mu']
decay = loaded_hawkes_data['decay']
tmax = loaded_hawkes_data['tmax']

print("Hawkes process parameters:")
for label, val in [("mu", mu), ("decay", decay), ("tmax", tmax)]:
    print("{:<15}{:<15}".format(label, val))

times_tensor, seq_types, seq_lengths = process_loaded_sequences(loaded_hawkes_data)
onehot_types = one_hot_embedding(seq_types, process_dim + 1)

hidden_size = 24
learning_rate = 0.015
model = HawkesDecayRNN(process_dim, hidden_size)
optimizer = optim.SGD(model.parameters(), learning_rate)

train_size = int(0.5 * times_tensor.size(1))
print("Train sample size: {:<15}".format(train_size))

# Define training data
train_times_tensor = times_tensor[:, :train_size]
train_onehot_types = onehot_types[:, :train_size]
train_seq_lengths = seq_lengths[:train_size]

# Training parameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

loss_hist = train_decayrnn(model, optimizer, train_times_tensor, train_onehot_types, train_seq_lengths,
                           tmax, BATCH_SIZE, EPOCHS, use_jupyter=False)

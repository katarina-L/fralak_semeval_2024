#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30/01/2024
Author: Katarina
"""


import pickle as p
import numpy as np
import random as rd
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch import optim

import semeval_network as se
from instances_class import Semeval_Data
from meme_labels_nn import get_network, prepare_data, show_plot, time_elapsed
from preprocess import preprocess_data


MEME_SIZE = 0
VOC_SIZE = 0
HIDDEN_SIZE = 128

MAX_STRING = 50

EOS = 1

class final_NN(nn.Module):
    def __init__(self):
        super(final_NN, self).__init__()

        self.meme_size = MEME_SIZE
        self.voc_size = VOC_SIZE
        self.max_size = MAX_STRING
        self.hidden_size = HIDDEN_SIZE
        self.in_shape = 32

        self.meme2b = nn.Linear(self.meme_size, self.in_shape)
        self.ml2b = nn.Linear(self.voc_size, self.in_shape)
        self.lrnn2b = nn.Linear(self.max_size, self.in_shape)

        self.in_combined = 32 * 3

        self.fc1 = nn.Linear(self.in_combined, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.voc_size)
        #self.fc3 = nn.Linear(self.hidden_size, self.voc_size)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, meme, ml, lrnn):
        lrnn = lrnn.float()
        meme = self.meme2b(meme)
        ml = self.ml2b(ml)
        lrnn = self.lrnn2b(lrnn)
        x = torch.cat((meme, ml, lrnn))

        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        x = self.dropout(x)
        out = self.relu(x)
        return(out)

def train_nn(model, data, n, n_epochs = 50, print_every = 5):
    start_time = time.time()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    all_losses = []

    for epoch in range(n_epochs):
        epoch_data = data.instances
        rd.shuffle(epoch_data)
        epoch_loss = 0
        for i, instance in enumerate(epoch_data):
            model.train()
            meme_embed = data.meme_embeds[instance['meme_index']]
            ml = instance['ml_output']
            ml = torch.Tensor(ml)
            lrnn = instance['lrnn_output']
            lrnn = torch.Tensor(lrnn)
            target_label = instance['meme_label']
            out = model(meme_embed, ml, lrnn)
            l = criterion(out, target_label)
            epoch_loss += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        all_losses.append(int(epoch_loss) / len(epoch_data))
        if (epoch+1) % print_every == 0:
            elapsed = time_elapsed(start_time, time.time())
            print(f'Epoch: {epoch+1}\tLoss: {epoch_loss:.3f}\tAverage: {epoch_loss / len(epoch_data):.3f}\tTime elapsed: {elapsed:.2f} min')

    print(f'\nSuccessfully trained for {n_epochs} epochs!\n')
    show_plot(all_losses, f'final_losses_{n}.png')

    # now save the model to disk
    f_name = f'final_model_{n}.pth'
    torch.save(model, f_name)
    print(f'Model saved to {f_name}')





def main():
    n = 5
    # network_path = 'network_with_embeds.p'
    # se_network = get_network(network_path)
    # data = prepare_data(se_network)
    with open('pickled_network2.p', 'rb') as f:
        data = p.load(f)
    #data = [(m['text'], m['labels'], m['id']) for m in data]
    #network = preprocess_data(data)

    global MEME_SIZE
    MEME_SIZE = len(data.meme_embeds[0])
    global VOC_SIZE
    VOC_SIZE = data.voc_size

    model = final_NN()
    print('NN initialized!')
    train_nn(model, data, n)

if __name__ == '__main__':
    main()
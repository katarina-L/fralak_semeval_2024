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

MEME_SIZE = 0
VOC_SIZE = 0
HIDDEN_SIZE = 128

EOS = 1

class label_RNN(nn.Module):
    # inspired by https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    def __init__(self):
        super(label_RNN, self).__init__()
        self.meme_size = MEME_SIZE
        self.voc_size = VOC_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.in_combined = self.voc_size + self.hidden_size

        self.m2h = nn.Linear(self.meme_size, self.hidden_size)
        self.i2h = nn.Linear(self.in_combined, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.voc_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden, meme=None):
        if meme is not None:
            hidden = self.m2h(meme)
        #print(f'Hidden: {hidden.size()}, input: {input.size()}')
        input_combined = torch.cat((hidden, input))
        #print(f'combined: {input_combined.size()}')
        hidden = self.i2h(input_combined)
        next_in = self.h2o(hidden)
        next_in = self.dropout(next_in)
        output = self.softmax(next_in)
        return(output, hidden, next_in)

    def initInput(self):
        return torch.zeros(self.voc_size)

    def initHidden(self):
        return torch.zeros(self.hidden_size)


def train_rnn(model, data, n, n_epochs = 25, print_every = 5):
    start_time = time.time()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.SmoothL1Loss()
    all_losses = []

    model.train()

    for epoch in range(n_epochs):
        epoch_data = list(data.instances)
        rd.shuffle(epoch_data)
        epoch_loss = 0

        for i, instance in enumerate(epoch_data):
            loss = torch.Tensor([0])
            hidden = model.initHidden()
            input = model.initInput()
            meme = data.meme_embeds[instance['meme_index']]
            for i_token, token in enumerate(instance['onehot']):
                if instance['label_indices'][i_token] == EOS:
                    break
                target = instance['onehot'][i_token+1]
                if i_token == 0:
                    output, hidden, input = model(input, hidden, meme = meme)
                else:
                    output, hidden, input = model(input, hidden)
                l = criterion(input, target)
                epoch_loss += l
                loss += l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_losses.append(int(epoch_loss) / len(epoch_data))

        if (epoch + 1) % print_every == 0:
            elapsed = time_elapsed(start_time, time.time())
            print(f'Epoch: {epoch + 1}\tLoss: {epoch_loss:.3f}\tAverage: {epoch_loss / len(epoch_data):.3f}\tTime elapsed: {elapsed:.2f} min')

    print(f'\nSuccessfully trained for {n_epochs} epochs!\n')
    show_plot(all_losses, f'all_lrnn_losses_{n}.png')

    f_name = f'rnn_label_model_{n}.pth'
    torch.save(model, f_name)
    print(f'Model saved to {f_name}')


def main():
    n = 4
    network_path = 'network_1_with_embeds_trainval_green.p'
    # se_network = get_network(network_path)
    # data = prepare_data(se_network)
    with open('pickled_data_trainval_green_4.p', 'rb') as f:
        data = p.load(f)

    global MEME_SIZE
    MEME_SIZE = len(data.meme_embeds[0])
    global VOC_SIZE
    VOC_SIZE = data.voc_size

    model = label_RNN()
    print('RNN initialized!')
    train_rnn(model, data, n)


if __name__ == '__main__':
    main()
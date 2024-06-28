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

MEME_SIZE = 0
VOC_SIZE = 0
HIDDEN_SIZE = 128

EOS = 1

class Meme_Labels(nn.Module):
    def __init__(self):
        super(Meme_Labels, self).__init__()

        self.meme_size = MEME_SIZE
        self.voc_size = VOC_SIZE
        self.hidden_size = HIDDEN_SIZE

        self.i2h = nn.Linear(self.meme_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.voc_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.i2h(x)
        out_no_relu = self.h2o(hidden)
        out = self.relu(out_no_relu)
        return(hidden, out_no_relu, out)


def show_plot(datapoints, figure_name):
    # based on showPlot function from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    plt.figure()
    plt.plot(datapoints)
    plt.savefig(figure_name)


def time_elapsed(start, end, as_minutes = True):
    # start and end are times
    elapsed = end - start
    if elapsed < 0:
        elapsed = 0
    if as_minutes:
        elapsed = elapsed / 60
    return(elapsed)


def get_network(path):
    with open(path, 'rb') as f:
        network = p.load(f)
    return (network)


def indices_to_tensor(indices, voc):
    tensor = torch.zeros(len(indices), len(voc))
    for i, word in enumerate(indices):
        tensor[i][word] = 1
    return(tensor)


def make_meme_tensor(indices, voc):
    tensor = torch.zeros(len(voc))
    for i in indices:
        tensor[i] = 1
    return(tensor)


def prepare_data(network, name = 'pickled_data.p'):
    # in: a se.network
    # out: a Semeval_Data object

    data = Semeval_Data()
    data.add_voc(network.label_lang.words)

    for i, l in enumerate(network.leaves):
        embed = l['embed']
        embed = torch.tensor(embed)
        embed = embed.squeeze()
        data.meme_embeds.append(embed)
        for label in l['restructured_labels']:
            instance = {}
            label_indices = network.label_lang.translate_words(label)
            onehot = indices_to_tensor(label_indices, data.voc)
            meme_label = make_meme_tensor(label_indices, data.voc)
            # instance = (i, onehot, label_indices, meme_label)
            instance['meme_index'] = i
            instance['onehot'] = onehot
            instance['label_indices'] = label_indices
            instance['meme_label'] = meme_label
            instance['labels'] = l['labels']
            data.instances.append(instance)

    print(f'We have a total of {len(data.instances)} instances, spread over {len(data.meme_embeds)} memes')
    with open(name, 'wb') as f:
        p.dump(data, f)
    print('pickle dumped!')

    return(data)


def train_nn(model, data, n, n_epochs = 45, print_every = 5):
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
            # loss = torch.Tensor([0])
            meme_embed = data.meme_embeds[instance['meme_index']]
            target_label = instance['meme_label']
            _, _, out = model(meme_embed)
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
    show_plot(all_losses, f'all_ml_losses_{n}.png')

    # now save the model to disk
    f_name = f'meme_label_model_{n}.pth'
    torch.save(model, f_name)
    print(f'Model saved to {f_name}')


def main():
    n = 4
    network_path = 'network_1_with_embeds_trainval_green.p'
    se_network = get_network(network_path)
    data = prepare_data(se_network, name = 'pickled_data_trainval_green_4.p')

    global MEME_SIZE
    MEME_SIZE = len(data.meme_embeds[0])
    global VOC_SIZE
    VOC_SIZE = data.voc_size
    print(VOC_SIZE)
    print(len(se_network.label_lang.words))
    print(se_network.label_lang.words)

    model = Meme_Labels()
    print('NN initialized!')
    train_nn(model, data, n)

if __name__ == '__main__':
    main()
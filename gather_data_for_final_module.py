#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30/01/2024
Author: Katarina
"""

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
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch import optim

from meme_labels_nn import Meme_Labels, make_meme_tensor, prepare_data
from label_rnn import label_RNN
import evaluate_models as ev

import semeval_network as se
from instances_class import Semeval_Data
from preprocess import preprocess_data

ML_THRESHOLD = 0.75
MAX_STRING = 50
EOS = 1


def add_ml_data(model_name, data, p_name = 'pickled_data_ml.p'):
    model = torch.load(model_name)
    model.eval()
    for i, inst in enumerate(data.instances):
        meme_index  = inst['meme_index']
        meme = data.meme_embeds[meme_index]
        _, _, output = model(meme)
        data.instances[i]['ml_output'] = output
    with open(p_name, 'wb') as f:
        p.dump(data, f)
    return(data)


def add_lrnn_data(model_name, data, p_name = 'val_pickled_lrnn.p'):
    model = torch.load(model_name)
    model.eval()
    voc = data.voc
    for i, inst in enumerate(data.instances):
        my_dict = data.instances[i]
        meme_index = inst['meme_index']
        # data.instances[i]['lrnn_output'] = []
        meme = data.meme_embeds[meme_index]
        preds = []
        input = model.initInput()
        hidden = model.initHidden()
        for i in range(MAX_STRING):
            if i == 0:
                output, hidden, input = model(input, hidden, meme=meme)
            else:
                output, hidden, input = model(input, hidden)
            _, output = output.topk(1)
            output = int(output)
            preds.append(output)
            if output == EOS:
                break
            else:
                input = torch.zeros(len(voc))
                input[output] = 1
        preds = torch.tensor(preds).float()
        preds = nn.functional.pad(preds, (0, MAX_STRING))
        preds = preds[:50]

        my_dict['lrnn_output'] = preds
        data.instances[i] = my_dict
        # data.instances[i]['lrnn_output'] = preds
        # data.voc_size = 34

    with open(p_name, 'wb') as f:
        p.dump(data, f)
    print('pickle dumped!')

def main():
    with open('network_2_with_embeds_trainval_green.p', 'rb') as f:
        data = p.load(f)
    data = prepare_data(data, name = 'pickled_network2.p')
    model_name = 'meme_label_model_4.pth'
    #with open('pickled_data_trainval_green_4.p', 'rb') as f:
    #    data = p.load(f)
    data = add_ml_data(model_name, data, p_name = 'n2_pickled_ml.p')
    add_lrnn_data('rnn_label_model_4.pth', data, p_name ='pickled_network2.p')

    #model_name = 'rnn_label_model_4.pth'
    #with open('pickled_ml_trainval_green_4.p', 'rb') as f:
    #    data = p.load(f)
    #add_lrnn_data(model_name, data, p_name = 'pickled_lrnn_trainval_green_4.p')
    '''
    # model_name = 'rnn_label_model_1.pth'
    # with open('pickled_data.p', 'rb') as f:
    #    data = p.load(f)
    with open('subtask1_data/validation.json', 'r') as f:
        data = json.load(f)
    # data = data[:50]
    data = process_data(data)

    #predictions, target_labels = evaluate_meme_label_nn(model_name, data)
    #predictions, target_labels = evaluate_label_rnn(model_name, data)

    #f1, recall, prec = se.get_metrics(predictions, target_labels)
    #print(f'F1: {f1:.2f} Recall: {recall:.2f} Precision: {prec:.2f}')
    '''

if __name__ == '__main__':
    main()
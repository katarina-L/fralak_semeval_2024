#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30/01/2024
Author: Katarina
"""
import os
import pickle as p
import re
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

from meme_labels_nn import Meme_Labels, make_meme_tensor
from final_module import final_NN
#from gather_data_for_final_module import add_ml_data, add_lrnn_data

import semeval_network as se
from instances_class import Semeval_Data
from preprocess import preprocess_data, preprocess_text
from label_rnn import label_RNN

from evaluate_models import process_data

EOS = 1
MAX_STRING = 50

FIN_THRESH = 0.4
PERS_THRESH = 1

def prepare_data(network, name = 'pickled_dev.p'):
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
            id = l['id']
            instance['id'] = id
        data.ids.append(id)


    print(f'We have a total of {len(data.instances)} instances, spread over {len(data.meme_embeds)} memes')
    with open(name, 'wb') as f:
        p.dump(data, f)
    print('pickle dumped!')


def prepare_dev_data(path, p_name):
    # pickles the dev data
    with open(path, 'r') as f:
        data = json.load(f)
    se_data = Semeval_Data()
    for meme in data:
        text = meme['text']
        embed, normalized_text = preprocess_text(text)
        embed = torch.tensor(embed)
        id = meme['id']
        se_data.meme_embeds.append(embed)
        se_data.ids.append(id)
    with open(p_name, 'wb') as f:
        p.dump(se_data, f)
    print('pickle dumped!')

def prepare_test_data():
    # pickles the test data
    test_files = os.listdir('test')
    test_data = []
    for filename in test_files:
        if re.search('\.json', filename) is not None:
            with open(f'test/{filename}', 'r') as f:
                data = json.load(f)
                test_data += data
                print(f'{filename} contains {len(data)} examples')
    print(f'Total of {len(test_data)} test examples!')
    se_data = Semeval_Data()
    for meme in test_data:
        text = meme['text']
        embed, normalized_text = preprocess_text(text)
        embed = torch.tensor(embed)
        id = meme['id']
        se_data.meme_embeds.append(embed)
        se_data.ids.append(id)
    print(len(se_data.ids))
    with open('test_pickled.p', 'wb') as f:
        p.dump(se_data, f)
    print('pickle dumped!')


def get_ml_output(model_name, p_name_in, p_name_out):
    with open(p_name_in, 'rb') as f:
        data = p.load(f)
    model = torch.load(model_name)
    model.eval()
    outputs = []
    for i, meme in enumerate(data.meme_embeds):
        _, _, output = model(meme)
        outputs.append(output)
    data.ml = outputs
    with open(p_name_out, 'wb') as f:
        p.dump(data, f)
    print('ML pickle dumped!')


def get_lrnn_output(model_name, p_name_in, p_name_out):
    with open(p_name_in, 'rb') as f:
        data = p.load(f)
    model = torch.load(model_name)
    model.eval()
    outputs = []
    network = se.Semeval_Network()
    voc_size = len(network.label_lang.words)
    for i, meme in enumerate(data.meme_embeds):
        preds = []
        input = model.initInput()
        hidden = model.initHidden()
        for i in range(MAX_STRING):
            if i == 0:
                output, hidden, input = model(input, hidden, meme = meme)
            else:
                output, hidden, input = model(input, hidden)
            _, output = output.topk(1)
            output = int(output)
            preds.append(output)
            if output == EOS:
                break
            else:
                input = torch.zeros(voc_size)
                input[output] = 1
        preds = torch.tensor(preds).float()
        preds = nn.functional.pad(preds, (0, MAX_STRING))
        preds = preds[:MAX_STRING]
        outputs.append(preds)
    data.lrnn = outputs
    with open(p_name_out, 'wb') as f:
        p.dump(data, f)
    print('lRNN pickle dumped!')
    return(data)


def get_final_output(model_name, p_name_in, p_name_out, try_times = 1):
    with open(p_name_in, 'rb') as f:
        data = p.load(f)
    model = torch.load(model_name)
    model.eval()
    outputs = []

    for n, meme in enumerate(data.meme_embeds):
        predictions = []
        ml = data.ml[n]
        lrnn = data.lrnn[n]
        for i in range(try_times):
            output = model(meme, ml, lrnn)
            predictions.append(output)
        outputs.append(predictions)

    data.finals = outputs
    with open(p_name_out, 'wb') as f:
        p.dump(data, f)
    print('final pickle dumped!')


def vote(preds):
    # in: a list of lists of predictions
    # out: an index corresponding list of final predictions
    final_preds = []
    for x in preds: # x = a list of lists containing predictions
        final_votes = [0] * len(x[0])
        for i, token in enumerate(final_votes):
            cum_score = 0
            for pred in x: # for each list of predictions
                cum_score += pred[i]
            score = cum_score / len(x)
            if score > FIN_THRESH:
                final_votes[i] = 1
            if i == 2: # persuasion
                if score < PERS_THRESH:
                    final_preds.append([0] * len(x[0]))
                    break
        final_preds.append(final_votes)
    return(final_preds)


def preds_to_labels(preds):
    # preds is a list of predictions
    # out: list of labels (index corresponding)
    network = se.Semeval_Network()
    voc = network.label_lang.words
    labels = []
    for m in preds:
        lab = [voc[i] for i, x in enumerate(m) if x > 0]
        no_daughters = []
        for label in lab:
            if label in network.nodes: # only check for real labels, not stuff like 'and' or 'SOS'
                # check if there are daughter nodes present; if so, we won't use it
                node = network.nodes[label]
                descendants = list(node.daughters)
                has_daughters = False
                while len(descendants) > 0:
                    daughters = network.nodes[descendants[0]].daughters
                    descendants += daughters
                    if descendants[0] in lab:
                        has_daughters = True
                    descendants.pop(0)
                if not has_daughters:
                    no_daughters.append(label)
        # no_daughters.sort(key=lambda x: network.labels_order.index(x))
        labels.append(no_daughters)
    return(labels)

def translate_dict():
    trans_dict = {'black': 'Black-and-white Fallacy/Dictatorship',
                      'logos': 'Logos',
                      'simplification': 'Simplification',
                      'loaded': 'Loaded Language',
                      'generalities': 'Glittering generalities (Virtue)',
                      'pathos': 'Pathos',
                      'ethos': 'Ethos',
                      'cliché': 'Thought-terminating cliché',
                      'whataboutism': 'Whataboutism',
                      'hominem': 'Ad Hominem',
                      'slogans': 'Slogans',
                      'distraction': 'Distraction',
                      'justification': 'Justification',
                      'reasoning': 'Reasoning',
                      'smears': 'Smears',
                      'oversimplification': 'Causal Oversimplification',
                      'names': 'Name calling/Labeling',
                      'authority': 'Appeal to authority',
                      'exaggeration': 'Exaggeration/Minimisation',
                      'repetition': 'Repetition',
                      'flag': 'Flag-waving',
                      'fear': 'Appeal to fear/prejudice',
                      'hitler': 'Reductio ad hitlerum',
                      'doubt': 'Doubt',
                      'scarecrow': 'Misrepresentation of Someone\'s Position (Straw Man)',
                      'vagueness': 'Obfuscation, Intentional vagueness, Confusion',
                      'bandwagon': 'Bandwagon',
                      'herring': 'Presenting Irrelevant Data (Red Herring)'}
    return(trans_dict)

def translate_labels(labels):
    trans_dict = translate_dict()
    reformed_labels = []
    for l in labels:
        if 'none' in l:
            reformed_labels.append([])
        else:
            l_reformed = []
            if 'persuasion' in l:
                l.remove('persuasion')
            for label in l:
                l_reformed.append(trans_dict[label])
            reformed_labels.append(l_reformed)
    return(reformed_labels)


def translate_semlabels(labels):
    t = translate_dict()
    t2 = {v : k for k, v in t.items()}
    reformed_labels = []
    for label in labels:
        ref_label = []
        for word in label:
            ref_label.append(t2[word])
        reformed_labels.append(ref_label)
    return(reformed_labels)


def explore_semlabels(labels):
    bg = labels[:436]
    ar = labels[436:536]
    en = labels[536:2036]
    mk = labels[2036:]
    print('yey')
    print(len(labels))
    for l in mk:
        print(l)

def vote_final_output(p_name_in, json_name_out, test = False):
    with open(p_name_in, 'rb') as f:
        data = p.load(f)

    predictions = vote(data.finals)
    print(len(data.ids))
    print(len(predictions))
    labels = preds_to_labels(predictions)
    sem_labels = translate_labels(labels)

    # explore_semlabels(sem_labels)

    if test:
        bg_output = []
        en_output = []
        mk_output = []
        ar_output = []


        for i, id in enumerate(data.ids):
            pred = {}
            pred['id'] = str(id)
            pred['labels'] = sem_labels[i]
            if re.search('(bg_)|(^[0-9]{,2}$)', str(id)) is not None:
                bg_output.append(pred)
            elif re.search('mk_', str(id)) is not None:
                mk_output.append(pred)
            elif re.search('^00', str(id)) is not None:
                ar_output.append(pred)
            else:
                en_output.append(pred)
        print(f'bg: {len(bg_output)} mk: {len(mk_output)} en: {len(en_output)} ar: {len(ar_output)}')

        with open(f'bg_{json_name_out}', 'w') as f:
            json.dump(bg_output, f)

        with open(f'en_{json_name_out}', 'w') as f:
            json.dump(en_output, f)

        with open(f'mk_{json_name_out}', 'w') as f:
            json.dump(mk_output, f)

        with open(f'ar_{json_name_out}', 'w') as f:
            json.dump(ar_output, f)
        
        print('jsons dumped!')

    else:
        with open('dev_subtask1_en.json', 'rb') as f:
            gold_labels = json.load(f)

        target_labels = [[]] * len(labels)

        for i, x in enumerate(gold_labels):
            id = x['id']
            labs = x['labels']
            index = data.ids.index(id)
            target_labels[index] = labs

        target_labels = translate_semlabels(target_labels)

        for i, label in enumerate(labels):
            if 'none' in label:
                labels[i] = []


        f1, recall, prec = se.get_metrics(labels, target_labels)
        print(f'F1: {f1:.2f} Recall: {recall:.2f} Precision: {prec:.2f}')


def main():
    data_type = 'dev7'
    #prepare_dev_data('subtask1_data/dev_unlabeled.json', f'{data_type}_pickled.p')
    #prepare_test_data()
    #get_ml_output('meme_label_model_4.pth', f'{data_type}_pickled.p', f'{data_type}_pickled_ml.p')
    #get_lrnn_output('rnn_label_model_4.pth', f'{data_type}_pickled_ml.p', f'{data_type}_pickled_lrnn.p')
    get_final_output('final_model_5.pth', f'{data_type}_pickled_lrnn.p', f'{data_type}_pickled_final.p', try_times = 1)
    vote_final_output(f'{data_type}_pickled_final.p', f'{data_type}_output_fralak.txt', test = False)


if __name__ == '__main__':
    main()
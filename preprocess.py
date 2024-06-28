#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13/12/2023
Author: Katarina
"""

import semeval_network as se
import json
import random as rd
import pickle as p
from sentence_transformers import SentenceTransformer

#import pandas as pd
#import numpy as np
import re
#import os

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langdetect import detect


rd.seed = 152
SMALL = False

rd.seed = 152

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=ner_model, tokenizer=tokenizer, grouped_entities=True)

embed_model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

def spelling(t):
    t = t.strip()
    t = re.sub(r'\\n', '\. ', t)
    t = re.sub(r'(.)\1{3,}', '\1\1', t)  # characters that repeat too much
    t = re.sub(r'[AaHhJjХхАа]*[HhJjХх]?[AaАа]+[HhJjХх]+[AaАа]+[AaHhJjХхАа]*', 'haha', t)  # haha-norm
    # t = re.sub('\'', '', t)
    t = re.sub('\W', ' ', t)
    t = re.sub('\s{2,}', ' ', t)
    t = re.sub('^ ', '', t)
    return (t)


def preprocess_text(text):
    # preprocess a text (text = string)
    # return an embedding
    ##TODO make a people dict with the most occurring people and their names
    people_dict = {'Trump': ['donald', 'donald trump', 'trump', 'donald j trump'],
                   'Biden': ['joe', 'joe biden', 'biden', 'bidens'],
                   'Putin': ['putin', 'vladimir putin', 'vladimir', 'vlad', 'vova', 'putins', 'vlads'],
                   'Obama': ['obama', 'barack obama', 'barack', 'obama\'s', 'hussein'],
                   'Clinton': ['clintons', 'hillary', 'hillary clinton', 'bill', 'bill clinton'],
                   # i count them as 1 person lol
                   'Reagan': ['ronald reagan', 'reagan', 'ronald'],
                   'Sanders': ['bernie sanders', 'bernie', 'sanders'],
                   'Pelosi': ['nancy', 'nancy pelosi', 'pelosi']
                   }
    lookup_dict = {}

    for p in people_dict:
        for alias in people_dict[p]:
            lookup_dict[alias] = p

    people_list = ['Trump', 'Biden', 'Putin', 'Obama', 'Clinton', 'Sanders', 'Reagan', 'Jesus',
                   'Pelosi']
    new_text = spelling(text)

    ner_results = nlp(new_text) # outputs a list of dictionaries
    for e in ner_results:
        if e['entity_group'] == 'PER' and e['score'] > 0.8:
            word = e['word']
            if word.lower() in lookup_dict:
                ref_name = lookup_dict[word.lower()]
                new_text = re.sub(word, ref_name, new_text)
            else:
                new_text = re.sub(word, 'Mark', new_text) # everyone is Mark now

    embedding = embed_model.encode(text)

    return(embedding, new_text.lower())


def get_data(path):
    f = open(path)
    data = json.load(f)
    # print(f'{type(data)}\n {len(data)}\n{type(data[0])}\n{data[0]}')
    data = [(m['text'], m['labels'], m['id']) for m in data]
    if SMALL:
        data = data[:50]
    return(data)


def preprocess_data(data):
    # initialize our network
    network = se.Semeval_Network()
    # add the data as leaves
    network.add_leaves(data)
    print('leaves added')
    # this restructures the labels to turn them into sequences of the desired format
    network.restructure_all_leaves()
    print('labels restructured')
    network.preprocess_leaves()
    print('preprocessing done')
    network.add_sentence_embeddings()
    print('embeddings added')
    return(network)

def preprocess_dev(data):
    # initialize network
    network = se.Semeval_Network()
    for d in data:
        print(d)
        break

def main():
    train_data = get_data('subtask1_data/train.json')
    val_data = get_data('subtask1_data/validation.json')
    train_data = train_data + val_data
    rd.shuffle(train_data)
    split = int(0.75 * len(train_data))
    train_1 = train_data[:split]
    train_2 = train_data[split:]
    print(len(train_1), len(train_2), len(train_data))

    # preprocess_dev(val_data)
    network_1 = preprocess_data(train_1)
    print('network 1 done')
    network_2 = preprocess_data(train_2)
    print('network 2 done')
    if SMALL:
        pickle_name = 'network_with_embeds_small.p'
    else:
        pickle_name = 'network_1_with_embeds_trainval_green.p'
    pickle_name = 'network_1_with_embeds_trainval_green.p'
    with open(pickle_name, 'wb') as f:
        p.dump(network_1, f)
    print(f'pickle {pickle_name} dumped')

    pickle_name = 'network_2_with_embeds_trainval_green.p'
    with open(pickle_name, 'wb') as f:
        p.dump(network_2, f)
    print(f'pickle {pickle_name} dumped!')


if __name__ == '__main__':
    main()
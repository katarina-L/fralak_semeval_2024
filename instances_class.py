#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30/01/2024
Author: Katarina
"""

class Semeval_Data():
    def __init__(self):
        # just lists of aligned things to save stuff
        self.meme_embeds = []
        self.ids = []           # this list is index-corresponding with meme_embeds
        self.ml = []            # same
        self.lrnn = []          # same
        self.finals = []        # same
        self.instances = []     # a list of dictionaries with the keys:
                                # 'meme_index' = the index in meme_embeds of the corresponding meme embedding
                                # 'labels' = the target labels
                                # 'label_indices' = the target labels, restructured, as voc indices
                                # 'label_onehot' = the labels as one hot vectors
                                # 'meme_label_target' = the restructured labels as a tensor of 1/0 for each voc word that appears
                                # 'id' = id
        self.voc = []

        self.voc_size = 0
        self.meme_size = 0
        self.hidden_size = 0

    def add_voc(self, voc):
        self.voc = voc
        self.voc_size = len(voc)

    def ttm(self):
        x = 1


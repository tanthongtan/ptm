# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:55:14 2020

@author: Tan
"""
import numpy as np

with open('diff.train.tfidf1.data') as file:
    for i, line in enumerate(file):
        tokens = line.split()
        if i == 0:
            vocab_size = int(tokens[0])
            num_tr = int(tokens[1])
            data_tr = np.zeros((num_tr, vocab_size))
        elif i == num_tr + 1:
            categories = float(tokens[0])
        else:
            for token in tokens[2:]:
                temp = token.split(':')
                idx = int(temp[0])
                val = float(temp[1])
                data_tr[i-1][idx] = val
                
vocab = {}
with open('diff.voc') as file:
    for i, line in enumerate(file):
        vocab[line.strip()] = i
        
    
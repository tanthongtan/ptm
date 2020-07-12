# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:52:20 2020

@author: Tan
"""
import collections
import numpy as np
from numbers import Number
import torch.nn.functional as F
import torch
import distributions as D
import torch.distributions as dist
from statistics import mean

def get_topic_uniqueness(top_words_idx_all_topics):
    """
    This function calculates topic uniqueness scores for a given list of topics.
    For each topic, the uniqueness is calculated as:  (\sum_{i=1}^n 1/cnt(i)) / n,
    where n is the number of top words in the topic and cnt(i) is the counter for the number of times the word
    appears in the top words of all the topics.
    :param top_words_idx_all_topics: a list, each element is a list of top word indices for a topic
    :return: a dict, key is topic_id (starting from 0), value is topic_uniquness score
    """
    n_topics = len(top_words_idx_all_topics)

    # build word_cnt_dict: number of times the word appears in top words
    word_cnt_dict = collections.Counter()
    for i in range(n_topics):
        word_cnt_dict.update(top_words_idx_all_topics[i])

    uniquenesses = []
    for i in range(n_topics):
        cnt_inv_sum = 0.0
        for ind in top_words_idx_all_topics[i]:
            cnt_inv_sum += 1.0 / word_cnt_dict[ind]
        uniquenesses.append(cnt_inv_sum / len(top_words_idx_all_topics[i]))
        
    mean_tu = mean(uniquenesses)

    return uniquenesses, mean_tu

def save_topics(topics, name):
    with open('topics_'+name+'.txt', 'w') as file:
        for i, line in enumerate(topics):
            for word in line:
                file.write(word+' ')
            if i < (len(topics)-1):
                file.write('\n')
                
def get_topics(topic_matrix, vocab, n_top_words = 10):
    topics = []
    for i, topic_dist in enumerate(topic_matrix):
        topic_words = np.array(list(dict(sorted(vocab.items(), key=lambda x:x[1])).keys()))[np.argsort(topic_dist)][:-n_top_words-1:-1]
        topics.append(list(topic_words))
    return topics

def vmf_perplexity(tensor_te, mu_final, kappa_final, alpha, N=1000):
    result = 0
    for i,doc_te in enumerate(tensor_te):
        prior_pi = dist.Dirichlet(alpha.flatten()).sample([N])
        if isinstance(kappa_final, Number):
            avg = kappa_final * F.normalize(torch.matmul(prior_pi,mu_final), p=2, dim=-1)
        else:
            avg = torch.matmul(prior_pi, kappa_final * mu_final)
        log_likelihood = D.log_prob_von_mises_fisher(avg, doc_te)
        result += torch.logsumexp(log_likelihood, -1) - np.log(N)
    return - 1. / tensor_te.shape[0] * result
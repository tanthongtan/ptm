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
import random
import subprocess
import os
import scipy.sparse
import sklearn.metrics
from lcdk import ratio
import math

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
        
    return uniquenesses, mean(uniquenesses)

def get_coherences(result):
    coherences = []
    for i, line in enumerate(result.strip().split('\n')):
        if i == 0:
            continue
        else:
            coherences.append(float(line.split()[1]))
    return coherences, mean(coherences)

def print_summary(topics, method, dataset, num_topic, M, num_samples):
    filename = str(random.randint(0,100000000))
    save_topics(topics,filename)
    result = subprocess.Popen(["java", "-jar", "palmetto-exec.jar", "wiki_final/wiki_final", "NPMI", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
    coherences_all, mean_coherence_all = get_coherences(result)
    uniquenesses_all = []
    print("\nMethod  =", method)
    print("Number of topics =", num_topic)
    print("Dataset =", dataset, "\n")
    for i in range(M):
        coherences_run = []
        uniquenesses_run = []
        for j in range(num_samples):
            sample_start_idx = i * num_samples * num_topic + j * num_topic
            sample_topic_indices = slice(sample_start_idx, sample_start_idx + num_topic)
            sample_topics = topics[sample_topic_indices]
            sample_coherences = coherences_all[sample_topic_indices]
            mean_coherence_sample = mean(sample_coherences)
            sample_uniquenesses, mean_uniqueness_sample = get_topic_uniqueness(sample_topics)
            
            print(" NPMI      ", "TU        ", "Topic") 
            for coherence, uniqueness, topic in zip(sample_coherences, sample_uniquenesses, sample_topics):
                print("{:8.5f} {:10.5f}   ".format(coherence, uniqueness), *topic)
            print("\nSample Mean NPMI =", mean_coherence_sample)
            print("Sample Mean TU   =", mean_uniqueness_sample, "\n")
            coherences_run.append(mean_coherence_sample)
            uniquenesses_run.append(mean_uniqueness_sample)
        mean_uniqueness_run = mean(uniquenesses_run)
        print("\nRun Mean NPMI =", mean(coherences_run))
        print("Run Mean TU   =", mean_uniqueness_run, "\n")
        uniquenesses_all.append(mean_uniqueness_run)
    print("\nAll Mean NPMI =", mean_coherence_all)
    print("All Mean TU   =", mean(uniquenesses_all), "\n")
    os.remove(filename)

def save_topics(topics, filename):
    with open(filename, 'w') as file:
        for topic in topics:
            print(*topic,file=file)
            
def print_topics(topics):
    for topic in topics:
        print(*topic)
                
def get_topics(topic_matrix, vocab, n_top_words = 10):
    topics = []
    for i, topic_dist in enumerate(topic_matrix):
        topic_words = np.array(list(dict(sorted(vocab.items(), key=lambda x:x[1])).keys()))[np.argsort(topic_dist)][:-n_top_words-1:-1]
        topics.append(list(topic_words))
    return topics

def vmf_perplexity(tensor_te, mu_final, kappa_final, alpha, N=1000):
    M = mu_final.shape[0]
    S = mu_final.shape[1]
    result = 0
    for i,doc_te in enumerate(tensor_te):
        prior_pi = dist.Dirichlet(alpha).sample([M, N]).unsqueeze(-3)
        if isinstance(kappa_final, Number):
            avg = kappa_final * F.normalize(torch.matmul(prior_pi,mu_final), p=2, dim=-1)
        else:
            avg = torch.matmul(prior_pi, kappa_final.unsqueeze(-1) * mu_final)
        log_likelihood = D.log_prob_von_mises_fisher_single_datapoint(avg, doc_te)
        result += torch.logsumexp(log_likelihood, dim=[-2,-1]) - np.log(N * S)
    return - 1. / tensor_te.shape[0] * result

def clustering_metrics_20news(pi):
    data_tr = scipy.sparse.load_npz("data/x_train_20news.npz")
    y_tr = np.load("data/y_train_20news.npy")
    
    chosen = data_tr.getnnz(1) > 0
    y_tr = y_tr[chosen]
    target = pi.argmax(axis=1)

    nmi = sklearn.metrics.normalized_mutual_info_score(y_tr,target)
    ari = sklearn.metrics.adjusted_rand_score(y_tr,target)
    ami = sklearn.metrics.adjusted_mutual_info_score(y_tr,target)
    flk = sklearn.metrics.fowlkes_mallows_score(y_tr,target)  
    hom = sklearn.metrics.homogeneity_score(y_tr, target)  
    print("                NMI:   %.4f" % nmi)
    print("Adjusted RAND index:   %.4f" % ari)
    print("        Adjusted MI:   %.4f" % ami)
    print("            Fowlkes:   %.4f" % flk)
    print("        Homogeneity:   %.4f" % hom)
    return nmi, ari, ami, flk, hom

def get_invalid_topics(pi, kappa, threshold = None):
    weights = pi*kappa.flatten()
    
    if threshold is None:
        threshold = 1/pi.shape[1]

    for i in range(weights.shape[0]):
        weights[i] = weights[i]/np.sum(weights[i])
    
    invalids = []
    for j in range(weights.shape[1]):
        invalid = True
        for i in range(weights.shape[0]):  
            if weights[i][j]>=threshold:
                invalid=False
                break
        if invalid:
            invalids.append(j)
    return invalids

def get_mrl(kappa, mu):
    return ratio(mu.shape[-1]/2, kappa)


def summarize_3d(var, var_name):
    mean = var.mean(dim=-1)
    print(f"{var_name} mean: {mean}")
    min, _ = var.min(dim=-1)
    print(f"{var_name} min: {min}")
    max, _ = var.max(dim=-1)
    print(f"{var_name} max: {max}")
    q = torch.tensor([0.05, 0.95])
    q5, q95 = var.quantile(q, dim=-1)
    print(f"{var_name} q5: {q5}")
    print(f"{var_name} q95: {q95}")
    return mean, min, max, q5, q95


def normalized_entropy(var):
    return -(var * var.clamp_min(1e-15).log()).sum(-1) / math.log(var.shape[-1])

def get_beta(t, L, endpoint=1):
    phase = ((t % L) + 1) / L
    return (1 + math.cos(2 * math.pi * phase / endpoint))/2 if phase <= endpoint else 1

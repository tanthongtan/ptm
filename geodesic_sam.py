# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:06:48 2020

@author: Tan
"""

import torch
import torch.nn.functional as F
from geodesic import SphericalGeodesicMonteCarlo
import dataset
import geodesic
import distributions as D
import time

torch.set_default_tensor_type(torch.cuda.FloatTensor)


"""
Load Data
"""
data_tr, tensor_tr, vocab, vocab_size, num_tr = dataset.load_20news_diff()
    
"""
Hyperparameters
"""
num_topic = 20
alpha = 0.5
c0 = 1000.0
kappa0 = 1000.0
kappa1 = 1000.0

alpha = torch.full((1,num_topic), alpha)
mu0 = F.normalize(tensor_tr.sum(dim=0),dim=-1)

pi = F.normalize(torch.randn(num_tr, num_topic), p=2, dim=-1)
mu = F.normalize(torch.randn(num_topic, vocab_size), p=2, dim=-1)


num_samples = 20000
num_burn = 0

kernel = SphericalGeodesicMonteCarlo(10, 1e-6)

for i in range(num_samples+num_burn):
    start_time = time.time()
    pi=kernel.transition(pi, D.SamFullConditionalPiDistribution(tensor_tr, mu, alpha, kappa1))
    mu = kernel.transition(mu, D.SamFullConditionalMuDistribution(tensor_tr, pi, c0, mu0, kappa0, kappa1))
    if i == num_burn:
        pi_samples = pi.unsqueeze(0)
        mu_samples = mu.unsqueeze(0)
    if i > num_burn:
        pi_samples = torch.cat((pi_samples,pi.unsqueeze(0)))
        mu_samples = torch.cat((mu_samples,mu.unsqueeze(0)))
    if i % 1 == 0:
        print(geodesic.accept_prob, i, time.time() - start_time)
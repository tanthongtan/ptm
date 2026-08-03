# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:06:48 2020

@author: Tan
"""

import torch
import numpy as np
from dataset import get_block_diag_data_batches_all_chains

def grad(f):
    def result(params):
        params_ = {name: param.detach().requires_grad_(True) for name, param in params.items()}
        f(params_).backward()
        return {name: param_.grad for name, param_ in params_.items()}
    return result
    

class GeodesicMonteCarlo:

    def __init__(self, M, data_tr, S, gpu = False):
        self.M = M
        self.data_tr = data_tr
        self.gpu = gpu
        self.S = S
        self.stochastic_gradient = S < data_tr.shape[0]

        if not self.stochastic_gradient:
            self.x, self.idx = get_block_diag_data_batches_all_chains(data_tr=data_tr, S=S, M=M, gpu=gpu)
        

    def stochastic_transition(self, params, vs, geodesics, distribution):
        params_star = {name: param.clone() for name, param in params.items()}
        vs_star = {name: v.clone() for name, v in vs.items()}
        for name, _ in params_star.items():
            params_star[name], vs_star[name] = geodesics[name].geodesic(params_star[name], vs_star[name])
            vs_star[name] = np.exp(-geodesics[name].c*geodesics[name].epsilon/2) * vs_star[name]
        if self.stochastic_gradient:
            x, idx = get_block_diag_data_batches_all_chains(data_tr=self.data_tr, S=self.S, M=self.M, gpu=self.gpu)
        else:
            x, idx = self.x, self.idx
        grads = grad(distribution.get_unnormalized_log_prob(x=x, idx=idx))(params_star)
        for name, _ in params_star.items():                
            vs_star[name] = geodesics[name].projection(params_star[name], vs_star[name] + grads[name]*geodesics[name].epsilon + (torch.randn_like(params_star[name]) * ((2*geodesics[name].c*geodesics[name].epsilon)**0.5)))
            vs_star[name] = np.exp(-geodesics[name].c*geodesics[name].epsilon/2) * vs_star[name]
            params_star[name], vs_star[name] = geodesics[name].geodesic(params_star[name], vs_star[name])                
        return params_star, vs_star

class Geodesic:
    
    def __init__(self, epsilon = None, lambda_param = 1, c=None, gamma=None, zeta=None, N=None):
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = lambda_param * np.sqrt(gamma/N)
        if c is not None:
            self.c = c
        elif zeta is not None:
            self.c = zeta/self.epsilon
        
    def projection(self, x, v):
        raise NotImplementedError
    
    def geodesic(self, x, v):
        raise NotImplementedError

class SphericalGeodesic(Geodesic):
    
    def projection(self, x, v):
        v = v - (x*v).sum(dim=-1).unsqueeze(-1) * x
        return v
    
    def geodesic(self, x, v):
        epsilon = self.epsilon / 2
        v_norm = v.norm(p=2, dim = -1).unsqueeze(-1)
        cos_norm_t = torch.cos(v_norm * epsilon) 
        sin_norm_t = torch.sin(v_norm * epsilon)
        x_new = x * cos_norm_t + v / v_norm * sin_norm_t
        v_new = v * cos_norm_t - v_norm * x * sin_norm_t
        return (x_new, v_new)
    
class PositiveGeodesic(Geodesic):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v):
        epsilon = self.epsilon / 2
        x_proposed = x + epsilon * v
        negative_indices = x_proposed < 0
        x_new = x_proposed.abs()
        v_new = torch.where(negative_indices, -v, v)
        return (x_new, v_new)
    
class RnGeodesic(Geodesic):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v):
        epsilon = self.epsilon / 2
        x_new = x + epsilon * v
        v_new = v
        return (x_new, v_new)
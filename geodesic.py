# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:06:48 2020

@author: Tan
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataset import get_block_diag_data_batches_all_chains

def grad(f):
    def result(params):
        params_ = {name: param.detach().requires_grad_(True) for name, param in params.items()}
        f(params_).backward()
        return {name: param_.grad for name, param_ in params_.items()}
    return result
    
def kinetic_per_chain(vs):
    return 0.5 * torch.cat([v.flatten(start_dim=1) for v in vs.values()], dim=-1).square().sum(dim=-1)

class GeodesicMonteCarlo:

    def __init__(self, M, data_tr, S, T = None, gpu = False):
        self.M = M
        self.data_tr = data_tr
        self.T = T
        self.gpu = gpu
        self.S = S
        self.stochastic_gradient = S < data_tr.shape[0]

        if not self.stochastic_gradient:
            self.x, self.idx = get_block_diag_data_batches_all_chains(data_tr=data_tr, S=S, M=M, gpu=gpu)


    def transition(self, params, geodesics, distribution):
        unnormalized_log_prob = distribution.get_unnormalized_log_prob(x=self.x, idx=self.idx)
        vs = {name: geodesics[name].projection(params[name],torch.randn_like(params[name])) for name in params}
        h = distribution.unnormalized_log_prob_per_chain(params, self.x, self.idx) - kinetic_per_chain(vs)
        params_star = {name: param.clone() for name, param in params.items()}
        grads = grad(unnormalized_log_prob)(params_star)
        for _ in range(self.T):
            for name, param_star in params_star.items():
                vs[name] = geodesics[name].projection(param_star, vs[name] + geodesics[name].epsilon/2.0 * grads[name])
                params_star[name], vs[name] = geodesics[name].geodesic(param_star, vs[name], 1)
            grads = grad(unnormalized_log_prob)(params_star)
            for name, param_star in params_star.items():
                vs[name] = geodesics[name].projection(param_star, vs[name] + geodesics[name].epsilon/2.0 * grads[name])
        h_star = distribution.unnormalized_log_prob_per_chain(params_star, self.x, self.idx) - kinetic_per_chain(vs)
        u = torch.rand_like(h_star)
        accept_prob = torch.exp(h_star - h)
        for name, param in params.items():
            params_star[name][u >= accept_prob, :] = param[u >= accept_prob, :]
        return params_star, accept_prob
    

    def stochastic_transition(self, params, vs, geodesics, distribution):
        params_star = {name: param.clone() for name, param in params.items()}
        vs_star = {name: v.clone() for name, v in vs.items()}
        for name, _ in params_star.items():
            params_star[name], vs_star[name] = geodesics[name].geodesic(params_star[name], vs_star[name], 0.5)
            vs_star[name] = np.exp(-geodesics[name].c*geodesics[name].epsilon/2) * vs_star[name]
        if self.stochastic_gradient:
            x, idx = get_block_diag_data_batches_all_chains(data_tr=self.data_tr, S=self.S, M=self.M, gpu=self.gpu)
        else:
            x, idx = self.x, self.idx
        grads = grad(distribution.get_unnormalized_log_prob(x=x, idx=idx))(params_star)
        for name, _ in params_star.items():                
            vs_star[name] = geodesics[name].projection(params_star[name], vs_star[name] + grads[name]*geodesics[name].epsilon + (torch.randn_like(params_star[name]) * ((2*geodesics[name].c*geodesics[name].epsilon)**0.5)))
            vs_star[name] = np.exp(-geodesics[name].c*geodesics[name].epsilon/2) * vs_star[name]
            params_star[name], vs_star[name] = geodesics[name].geodesic(params_star[name], vs_star[name], 0.5)                
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
    
    def geodesic(self, x, v, time_multiplier):
        raise NotImplementedError

class SphericalGeodesic(Geodesic):
    
    def projection(self, x, v):
        v = v - (x*v).sum(dim=-1).unsqueeze(-1) * x
        return v
    
    def geodesic(self, x, v, time_multiplier):
        epsilon = self.epsilon * time_multiplier
        v_norm = v.norm(p=2, dim = -1).unsqueeze(-1)
        cos_norm_t = torch.cos(v_norm * epsilon) 
        sin_norm_t = torch.sin(v_norm * epsilon)
        x_new = x * cos_norm_t + v / v_norm * sin_norm_t
        v_new = v * cos_norm_t - v_norm * x * sin_norm_t
        x_new = F.normalize(x_new, p=2, dim=-1)
        v_new = self.projection(x_new, v_new)
        return (x_new, v_new)
    
class PositiveGeodesic(Geodesic):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v, time_multiplier):
        epsilon = self.epsilon * time_multiplier
        x_proposed = x + epsilon * v
        negative_indices = x_proposed < 0
        x_new = x_proposed.abs()
        v_new = torch.where(negative_indices, -v, v)
        return (x_new, v_new)
    
class RnGeodesic(Geodesic):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v, time_multiplier):
        epsilon = self.epsilon * time_multiplier
        x_new = x + epsilon * v
        v_new = v
        return (x_new, v_new)
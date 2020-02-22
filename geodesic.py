# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:06:48 2020

@author: Tan
"""

import torch
import torch.distributions as dist

def grad(f, independent_axes):
    def result(x):
        if independent_axes == 0:
            output_shape = torch.tensor(1.)
        elif independent_axes == 1:
            output_shape = torch.ones(x.shape[0])
        x_ = x.detach().requires_grad_(True) 
        f(x_).backward(output_shape)
        return x_.grad
    return result
    
class GeodesicMonteCarlo:
    
    def __init__(self, T = 20, eta = 1e-2):
        self.T = T
        self.eta = eta 
    
    def transition(self, x, distribution):
        global accept_prob
        v = dist.MultivariateNormal(torch.zeros(x.shape[-1]), torch.eye(x.shape[-1])).sample([x.shape[0]])
        v = self.projection(x, v)
        h = distribution.unnormalized_log_prob(x) - 0.5 * (v*v).sum(dim=-1)
        x_star = x.clone()
        for _ in range(self.T):
            v = v + self.eta/2.0 * grad(distribution.unnormalized_log_prob, distribution.independent_axes)(x_star)
            v = self.projection(x_star, v)
            x_star, v = self.geodesic(x_star, v)
            v = v + self.eta/2.0 * grad(distribution.unnormalized_log_prob, distribution.independent_axes)(x_star)
            v = self.projection(x_star, v)
        h_star = distribution.unnormalized_log_prob(x_star) - 0.5 * (v*v).sum(dim=-1)
        u = torch.rand_like(h_star)
        accept_prob = torch.exp(h_star - h)
        x_star[u >= accept_prob, :] = x[u >= accept_prob, :]
        return x_star
        
    def projection(self, x, v):
        raise NotImplementedError
    
    def geodesic(self, x, v):
        raise NotImplementedError

class SphericalGeodesicMonteCarlo(GeodesicMonteCarlo):
    
    def projection(self, x, v):
        proj_matrix = torch.bmm(x.unsqueeze(-1), x.unsqueeze(1))
        v = v.unsqueeze(-1) - torch.bmm(proj_matrix, v.unsqueeze(-1)) 
        v = v.squeeze(-1)
        return v
    
    def geodesic(self, x, v):
        v_norm = v.norm(p=2, dim = -1).unsqueeze(-1)
        cos_norm_t = torch.cos(v_norm * self.eta) 
        sin_norm_t = torch.sin(v_norm * self.eta)
        x_new = x * cos_norm_t + v / v_norm * sin_norm_t
        v_new = v * cos_norm_t - v_norm * x * sin_norm_t
        return (x_new, v_new)
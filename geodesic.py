# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:06:48 2020

@author: Tan
"""

import torch
import torch.distributions as dist

def grad(f, independent_axes):
    def result(x):
        if independent_axes == None:
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
        v = dist.MultivariateNormal(torch.zeros(x.shape[-1]), torch.eye(x.shape[-1])).sample([x.shape[0]])
        v = self.projection(x, v)
        h = distribution.unnormalized_log_prob(x) - 0.5 * v.norm(dim=distribution.independent_axes)**2
        x_star = x.clone()
        for _ in range(self.T):
            v = v + self.eta/2.0 * grad(distribution.unnormalized_log_prob, distribution.independent_axes)(x_star)
            v = self.projection(x_star, v)
            x_star, v = self.geodesic(x_star, v)
            v = v + self.eta/2.0 * grad(distribution.unnormalized_log_prob, distribution.independent_axes)(x_star)
            v = self.projection(x_star, v)
        h_star = distribution.unnormalized_log_prob(x_star) - 0.5 * v.norm(dim=distribution.independent_axes)**2
        u = torch.rand_like(h_star)
        accept_prob = torch.exp(h_star - h)
        x_star[u >= accept_prob, :] = x[u >= accept_prob, :]
        return x_star, accept_prob
        
    def projection(self, x, v):
        raise NotImplementedError
    
    def geodesic(self, x, v):
        raise NotImplementedError

class SphericalGeodesicMonteCarlo(GeodesicMonteCarlo):
    
    def projection(self, x, v):
        v = v - (x*v).sum(dim=-1).unsqueeze(-1) * x
        return v
    
    def geodesic(self, x, v):
        v_norm = v.norm(p=2, dim = -1).unsqueeze(-1)
        cos_norm_t = torch.cos(v_norm * self.eta) 
        sin_norm_t = torch.sin(v_norm * self.eta)
        x_new = x * cos_norm_t + v / v_norm * sin_norm_t
        v_new = v * cos_norm_t - v_norm * x * sin_norm_t
        return (x_new, v_new)
    
class PositiveHamiltonianMonteCarlo(GeodesicMonteCarlo):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v):
        x_new = x + self.eta * v
        v_new = v
        x_new[x_new<0] = -x_new[x_new<0]
        v_new[x_new<0] = -v_new[x_new<0]
        return (x_new, v_new)
    
class HamiltonianMonteCarlo(GeodesicMonteCarlo):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v):
        x_new = x + self.eta * v
        v_new = v
        return (x_new, v_new)
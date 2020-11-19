# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:06:48 2020

@author: Tan
"""

import torch
import torch.distributions as dist
import numpy as np

def grad(f):
    def result(params):
        params_ = {name: param.detach().requires_grad_(True) for name, param in params.items()}
        f(params_).backward()
        return {name: param_.grad for name, param_ in params_.items()}
    return result
    
class GeodesicMonteCarlo:
    
    def __init__(self, T = 20):
        self.T = T
    
    def transition(self, params, geodesics, distribution):
        vs = {}
        for name, param in params.items():
            vs[name] = dist.MultivariateNormal(torch.zeros(param.shape[-1]), torch.eye(param.shape[-1])).sample([param.shape[0]])
            vs[name] = geodesics[name].projection(param, vs[name])
        h = distribution.unnormalized_log_prob(params) - 0.5 * torch.cat([v.flatten() for v in vs.values()]).norm()**2
        params_star = {name: param.clone() for name, param in params.items()}
        for _ in range(self.T):
            grads = grad(distribution.unnormalized_log_prob)(params_star)
            for name, param_star in params_star.items():
                vs[name] = vs[name] + geodesics[name].eta/2.0 * grads[name]
                vs[name] = geodesics[name].projection(param_star, vs[name])
                params_star[name], vs[name] = geodesics[name].geodesic(param_star, vs[name])
            grads = grad(distribution.unnormalized_log_prob)(params_star)
            for name, param_star in params_star.items():
                vs[name] = vs[name] + geodesics[name].eta/2.0 * grads[name]
                vs[name] = geodesics[name].projection(param_star, vs[name])
        h_star = distribution.unnormalized_log_prob(params_star) - 0.5 * torch.cat([v.flatten() for v in vs.values()]).norm()**2
        u = torch.rand_like(h_star)
        accept_prob = torch.exp(h_star - h)
        for name, param in params.items():
            params_star[name][u >= accept_prob, :] = param[u >= accept_prob, :]
        return params_star, accept_prob
    
    def stochastic_transition(self, params, vs, geodesics, distribution):
        params_star = {name: param.clone() for name, param in params.items()}
        vs_star = {name: v.clone() for name, v in vs.items()}
        for _ in range(self.T):
            grads = grad(distribution.unnormalized_log_prob)(params_star)
            for name, param_star in params_star.items():
                params_star[name], vs_star[name] = geodesics[name].geodesic(param_star, vs_star[name])
                vs_star[name] = np.exp(-geodesics[name].c*geodesics[name].eta/2) * vs_star[name]
                vs_star[name] = geodesics[name].projection(param_star, vs_star[name] + grads[name]*geodesics[name].eta+dist.MultivariateNormal(torch.zeros(param_star.shape[-1]), 2*geodesics[name].c*geodesics[name].eta*torch.eye(param_star.shape[-1])).sample([param_star.shape[0]]))
                vs_star[name] = np.exp(-geodesics[name].c*geodesics[name].eta/2) * vs_star[name]
                params_star[name], vs_star[name] = geodesics[name].geodesic(param_star, vs_star[name])                
        return params_star, vs_star

class Geodesic:
    
    def __init__(self, eta = 1e-2, c=None, gamma=None, rho=None, N=None):
        if eta is not None:
            self.eta = eta
        else:
            self.eta = np.sqrt(gamma/N)
        if c is not None:
            self.c = c
        elif rho is not None:
            self.c = rho/self.eta
        
    def projection(self, x, v):
        raise NotImplementedError
    
    def geodesic(self, x, v):
        raise NotImplementedError

class SphericalGeodesic(Geodesic):
    
    def projection(self, x, v):
        v = v - (x*v).sum(dim=-1).unsqueeze(-1) * x
        return v
    
    def geodesic(self, x, v):
        eta = self.eta / 2
        v_norm = v.norm(p=2, dim = -1).unsqueeze(-1)
        cos_norm_t = torch.cos(v_norm * eta) 
        sin_norm_t = torch.sin(v_norm * eta)
        x_new = x * cos_norm_t + v / v_norm * sin_norm_t
        v_new = v * cos_norm_t - v_norm * x * sin_norm_t
        return (x_new, v_new)
    
class PositiveGeodesic(Geodesic):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v):
        eta = self.eta / 2
        x_new = x + eta * v
        v_new = v
        x_new[x_new<0] = -x_new[x_new<0]
        v_new[x_new<0] = -v_new[x_new<0]
        return (x_new, v_new)
    
class RnGeodesic(Geodesic):
    
    def projection(self, x, v):
        return v
    
    def geodesic(self, x, v):
        eta = self.eta / 2
        x_new = x + eta * v
        v_new = v
        return (x_new, v_new)
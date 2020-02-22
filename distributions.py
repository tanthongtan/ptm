import torch
from lcdk import Logcdk
import torch.nn.functional as F

class SamFullConditionalPiDistribution:
    
    def __init__(self, x, mu, alpha, kappa1):
        self.x = x
        self.mu = mu
        self.alpha = alpha
        self.kappa1 = kappa1
        self.independent_axes = 1

    def unnormalized_log_prob_spherical_dirichlet(self, pi):
        return ((2*self.alpha-1)*torch.log(torch.abs(pi))).sum(dim=-1)
    
    def unnormalized_log_prob(self, pi):
        avg = F.normalize(torch.matmul(pi,self.mu), p=2, dim=-1)
        return self.unnormalized_log_prob_spherical_dirichlet(pi) \
                + (self.kappa1 * avg * self.x).sum(dim=-1)
            

class SamFullConditionalMuDistribution:
    
    def __init__(self, x, pi, c0, mu0, kappa0, kappa1):
        self.x = x
        self.pi = pi
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        self.independent_axes = 0
    
    def unnormalized_log_prob(self, mu):
        logcdk = Logcdk.apply
        avg = F.normalize(torch.matmul(self.pi,mu), p=2, dim=-1)
        return (self.kappa1 * avg * self.x).sum(dim=-1).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1))
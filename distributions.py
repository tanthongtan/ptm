import torch
from lcdk import Logcdk
import torch.nn.functional as F

def unnormalized_log_prob_spherical_dirichlet(alpha, theta):
    return ((2*alpha-1)*torch.log(torch.abs(theta))).sum(dim=-1)

def log_prob_von_mises_fisher(D, natural_param, X):
    logcdk = Logcdk.apply
    return logcdk(D, natural_param.norm(p=2, dim=-1)) + (natural_param * X).sum(dim = -1)

class SamFullConditionalPiDistribution:
    
    def __init__(self, x, mu, alpha, kappa1):
        self.x = x
        self.mu = mu
        self.alpha = alpha
        self.kappa1 = kappa1
        self.independent_axes = 1

    def unnormalized_log_prob(self, theta):
        pi = theta**2
        avg = F.normalize(torch.matmul(pi,self.mu), p=2, dim=-1)
        return unnormalized_log_prob_spherical_dirichlet(self.alpha, theta) \
                + (self.kappa1 * avg * self.x).sum(dim=-1)
            

class SamFullConditionalMuDistribution:
    
    def __init__(self, x, theta, c0, mu0, kappa0, kappa1):
        self.x = x
        self.theta = theta
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        self.independent_axes = 0
    
    def unnormalized_log_prob(self, mu):
        logcdk = Logcdk.apply
        pi = self.theta ** 2
        avg = F.normalize(torch.matmul(pi,mu), p=2, dim=-1)
        return (self.kappa1 * avg * self.x).sum(dim=-1).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1))
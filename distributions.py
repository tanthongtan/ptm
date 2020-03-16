import torch
from lcdk import Logcdk
import torch.nn.functional as F
import torch.distributions as dist

def unnormalized_log_prob_spherical_dirichlet(alpha, theta):
    return ((2*alpha-1)*torch.log(torch.abs(theta))).sum(dim=-1)

def log_prob_stickbreaking_dirichlet(alpha, theta, pi):
    return dist.Dirichlet(alpha).log_prob(pi) - dist.StickBreakingTransform().inv.log_abs_det_jacobian(pi, theta)


def log_prob_von_mises_fisher(D, natural_param, X):
    logcdk = Logcdk.apply
    return logcdk(D, natural_param.norm(p=2, dim=-1)) + (natural_param * X).sum(dim = -1)

class SamFullConditionalThetaDistribution:
    
    def __init__(self, x, mu, alpha, kappa1):
        self.x = x
        self.mu = mu
        self.alpha = alpha
        self.kappa1 = kappa1
        self.independent_axes = 1

    def unnormalized_log_prob(self, theta):
        pi = dist.StickBreakingTransform()(theta)
        avg = F.normalize(torch.matmul(pi,self.mu), p=2, dim=-1)
        return log_prob_stickbreaking_dirichlet(self.alpha, theta, pi) \
                + (self.kappa1 * avg * self.x).sum(dim=-1)
            

class SamFullConditionalMuDistribution:
    
    def __init__(self, x, theta, c0, mu0, kappa0, kappa1):
        self.x = x
        self.theta = theta
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        self.independent_axes = None
    
    def unnormalized_log_prob(self, mu):
        logcdk = Logcdk.apply
        pi = dist.StickBreakingTransform()(self.theta)
        avg = F.normalize(torch.matmul(pi,mu), p=2, dim=-1)
        return (self.kappa1 * avg * self.x).sum(dim=-1).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1))
                
class VptmFullConditionalThetaDistribution:
    
    def __init__(self, x, mu, kappa, alpha):
        self.x = x
        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.independent_axes = 1
        
    def unnormalized_log_prob(self, theta):
        pi = dist.StickBreakingTransform()(theta)
        avg = torch.matmul(pi, self.kappa * self.mu)
        return log_prob_stickbreaking_dirichlet(self.alpha, theta, pi) \
                + log_prob_von_mises_fisher(self.mu.shape[-1], avg, self.x)
                
class VptmFullConditionalMuDistribution:
    
    def __init__(self, x, theta, kappa, c0, mu0, kappa0):
        self.x = x
        self.theta = theta
        self.kappa = kappa
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.independent_axes = None
        
    def unnormalized_log_prob(self, mu):
        logcdk = Logcdk.apply
        pi = dist.StickBreakingTransform()(self.theta)
        avg = torch.matmul(pi, self.kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1))
                
class VptmFullConditionalKappaDistribution:
    
    def __init__(self, x, theta, mu, m, sigma_squared):
        self.x = x
        self.theta = theta
        self.mu = mu
        self.m = m
        self.sigma_squared = sigma_squared
        self.independent_axes = None
        
    def unnormalized_log_prob(self, kappa):
        pi = dist.StickBreakingTransform()(self.theta)
        avg = torch.matmul(pi, kappa * self.mu)
        return log_prob_von_mises_fisher(self.mu.shape[-1], avg, self.x).sum() + \
                dist.LogNormal(self.m, self.sigma_squared).log_prob(kappa).sum()
                
class VptmStochasticFullConditionalMuKappaDistribution:
    
    def __init__(self, x, theta, c0, mu0, kappa0, m, sigma_squared):
        self.x = x
        self.theta = theta
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.m = m
        self.sigma_squared = sigma_squared
        self.independent_axes = None
        
    def unnormalized_log_prob(self, mu, kappa):
        logcdk = Logcdk.apply
        pi = dist.StickBreakingTransform()(self.theta)
        avg = torch.matmul(pi, (kappa * mu).unsqueeze(0))
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, (self.x).unsqueeze(0)).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1))\
                 + dist.LogNormal(self.m, self.sigma_squared).log_prob(kappa).sum()
                
class MptmFullConditionalThetaDistribution:

    def __init__(self, x, lamb, alpha):
        self.x = x
        self.lamb = lamb
        self.alpha = alpha
        
    def unnormalized_log_prob(self, theta):
        pi = dist.StickBreakingTransform()(theta)
        beta = torch.distributions.StickBreakingTransform()(self.lamb)
        avg = torch.exp(torch.matmul(pi, torch.log(beta)))
        return dist.Multinomial(probs = avg).log_prob(self.x) \
                + dist.Dirichlet(self.alpha).log_prob(pi)

class MptmFullConditionalLambDistribution:

    def __init__(self, x, theta, eta):
        self.x = x
        self.theta = theta
        self.eta = eta
        
    def unnormalized_log_prob(self, lamb):
        pi = dist.StickBreakingTransform()(self.theta)
        beta = torch.distributions.StickBreakingTransform()(lamb)
        avg = torch.exp(torch.matmul(pi, torch.log(beta)))
        return dist.Multinomial(probs = avg).log_prob(self.x).sum() \
                + dist.Dirichlet(self.eta).log_prob(beta).sum()
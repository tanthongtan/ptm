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

def log_prob_von_mises_fisher_polar(x, mu, kappa):
    logcdk = Logcdk.apply
    return logcdk(mu.shape[-1], kappa) + kappa * (mu * x).sum(dim=-1)

def log_prob_vmf_conjugate_prior(c, v, mu0, mu, kappa):
    logcdk = Logcdk.apply
    return v * logcdk(mu0.shape[-1], kappa) + c * kappa * (mu0 * mu).sum(dim=-1)


class SamJointDistributionWithStickDir:
    
    def __init__(self, x, alpha, c0, mu0, kappa0, kappa1):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        
    def unnormalized_log_prob(self, params):
        logcdk = Logcdk.apply
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        mu = params['mu']
        avg = F.normalize(torch.matmul(pi,mu), p=2, dim=-1)
        return (self.kappa1 * avg * self.x).sum(dim=-1).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1)) \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
class SamJointDistributionWithSphereDir:
    
    def __init__(self, x, alpha, c0, mu0, kappa0, kappa1):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        
    def unnormalized_log_prob(self, params):
        logcdk = Logcdk.apply
        theta = params['theta']
        pi = theta ** 2.
        mu = params['mu']
        avg = F.normalize(torch.matmul(pi,mu), p=2, dim=-1)
        return (self.kappa1 * avg * self.x).sum(dim=-1).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1)) \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, theta).sum()
                    
class VptmJointDistributionWithStickDir:
    
    def __init__(self, x, alpha, c0, mu0, kappa0, m, sigma_squared):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.m = m
        self.sigma_squared = sigma_squared
        
    def unnormalized_log_prob(self, params):
        logcdk = Logcdk.apply
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1)) \
                + dist.LogNormal(self.m, self.sigma_squared).log_prob(kappa).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
    def log_likelihood(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum()
    
class VptmJointDistributionWithStickDirGamma:
    
    def __init__(self, x, alpha, c0, mu0, kappa0, a, b):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.a = a
        self.b = b
        
    def unnormalized_log_prob(self, params):
        logcdk = Logcdk.apply
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1)) \
                + dist.Gamma(self.a, self.b).log_prob(kappa).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
    def log_likelihood(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum()
    
class VptmJointDistributionWithStickDirGammaVMFPrior:
    
    def __init__(self, x, alpha, c0, mu0, a, b):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.a = a
        self.b = b
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum() \
                + log_prob_von_mises_fisher_polar(mu, self.mu0, self.c0 * kappa).sum() \
                + dist.Gamma(self.a, self.b).log_prob(kappa).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
    def log_likelihood(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum()
    
class VptmJointDistributionWithStickDirConjugatePrior:
    
    def __init__(self, x, alpha, c, mu0, v, positive = False):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.positive = positive
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        if self.positive == True:
            mu = torch.abs(mu)
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum() \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
    def log_likelihood(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum()
    
class VptmJointDistributionWithSphereDir:
    
    def __init__(self, x, alpha, c0, mu0, kappa0, m, sigma_squared):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.m = m
        self.sigma_squared = sigma_squared
        
    def unnormalized_log_prob(self, params):
        logcdk = Logcdk.apply
        theta = params['theta']
        pi = theta ** 2.
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1)) \
                + dist.LogNormal(self.m, self.sigma_squared).log_prob(kappa).sum() \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, theta).sum()
                
    def log_likelihood(self, params):
        theta = params['theta']
        pi = theta ** 2.
        kappa = params['kappa']
        mu = params['mu']
        avg = torch.matmul(pi, kappa * mu)
        return log_prob_von_mises_fisher(self.mu0.shape[-1], avg, self.x).sum()
                                              
class MptmJointDistributionWithStickDir:
    
    def __init__(self, x, alpha, eta):
        self.x = x 
        self.alpha = alpha
        self.eta = eta
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        lamb = params['lamb']
        beta = torch.distributions.StickBreakingTransform()(lamb)
        avg = torch.exp(torch.matmul(pi, torch.log(beta)))
        return dist.Multinomial(probs = avg).log_prob(self.x).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum() \
                + log_prob_stickbreaking_dirichlet(self.eta, lamb, beta).sum()
                
class MptmJointDistributionWithSphereDir:
    
    def __init__(self, x, alpha, eta):
        self.x = x 
        self.alpha = alpha
        self.eta = eta
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = theta ** 2.
        lamb = params['lamb']
        beta = lamb ** 2.
        avg = torch.exp(torch.matmul(pi, torch.log(beta)))
        return dist.Multinomial(probs = avg).log_prob(self.x).sum() \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, theta).sum() \
                + unnormalized_log_prob_spherical_dirichlet(self.eta, lamb).sum()
                
class LdaJointDistributionWithStickDir:
    
    def __init__(self, x, alpha, eta):
        self.x = x 
        self.alpha = alpha
        self.eta = eta
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        lamb = params['lamb']
        beta = torch.distributions.StickBreakingTransform()(lamb)
        avg = torch.matmul(pi, beta)
        return dist.Multinomial(probs = avg).log_prob(self.x).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum() \
                + log_prob_stickbreaking_dirichlet(self.eta, lamb, beta).sum()  
                
class LdaJointDistributionWithSphereDir:
    
    def __init__(self, x, alpha, eta):
        self.x = x 
        self.alpha = alpha
        self.eta = eta
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = theta ** 2.
        lamb = params['lamb']
        beta = lamb ** 2.
        avg = torch.matmul(pi, beta)
        return dist.Multinomial(probs = avg).log_prob(self.x).sum() \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, theta).sum() \
                + unnormalized_log_prob_spherical_dirichlet(self.eta, lamb).sum()  
                
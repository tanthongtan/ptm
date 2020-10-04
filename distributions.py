import torch
from lcdk import Logcdk
import torch.nn.functional as F
import torch.distributions as dist

def log_prob_stickbreaking_dirichlet(alpha, theta, pi):
    return dist.Dirichlet(alpha).log_prob(pi) - dist.StickBreakingTransform().inv.log_abs_det_jacobian(pi, theta)

def log_prob_von_mises_fisher(natural_param, X):
    logcdk = Logcdk.apply
    if len(X.shape) == 1:
        dot = (natural_param * X.to_dense()).sum(dim = -1)
    else:
        dot = sparse_dense_dot(X,natural_param)
    return logcdk(natural_param.shape[-1], natural_param.norm(p=2, dim=-1)) + dot

def log_prob_von_mises_fisher_mix(mu, kappa, pi, X):
    logcdk = Logcdk.apply
    dot = torch.sparse.mm(X, torch.transpose(kappa * mu, 0, 1))
    log_norm = torch.transpose(logcdk(mu.shape[-1], kappa), 0, 1)
    log_pi = torch.log(pi)
    return torch.logsumexp(log_pi + log_norm + dot, -1)

def log_prob_vmf_conjugate_prior(c, v, mu0, mu, kappa):
    logcdk = Logcdk.apply
    return v * logcdk(mu0.shape[-1], kappa) + c * kappa * (mu0 * mu).sum(dim=-1)

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())

def sparse_dense_dot(s, d):
    return torch.sparse.sum(sparse_dense_mul(s, d),dim=1).to_dense()

class SamJointDistributionWithStickDirHyperprior:
    
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
        return self.kappa1 * sparse_dense_dot(self.x,avg).sum() \
                - logcdk(self.mu0.shape[-1], (self.kappa0 * self.mu0 + self.c0 * mu.sum(dim=0)).norm(p=2, dim=-1)) \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
class SamJointDistributionWithStickDir:
    
    def __init__(self, x, alpha, c0, mu0, kappa1):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa1 = kappa1
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        mu = params['mu']
        avg = F.normalize(torch.matmul(pi,mu), p=2, dim=-1)
        return self.kappa1 * sparse_dense_dot(self.x,avg).sum() \
                + self.c0 * (self.mu0 * mu).sum(dim=-1).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                                        
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
        return log_prob_von_mises_fisher(avg, self.x).sum() \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
                
class BvmfmixJointDistributionWithStickDirConjugatePrior:
    
    def __init__(self, x, alpha, c, mu0, v):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        kappa = params['kappa']
        mu = params['mu']
        return log_prob_von_mises_fisher_mix(mu, kappa, pi, self.x).sum() \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()
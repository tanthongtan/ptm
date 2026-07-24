import torch
from lcdk import Logcdk, ratio
import torch.nn.functional as F
import torch.distributions as dist
import math

def log_prob_stickbreaking_dirichlet(alpha, theta, pi):
    return dist.Dirichlet(alpha).log_prob(pi) - dist.StickBreakingTransform().inv.log_abs_det_jacobian(pi, theta)

def log_prob_ilr_dirichlet(alpha, theta, pi):
    return dist.Dirichlet(alpha).log_prob(pi) + HelmertILRTransform().inv.log_abs_det_jacobian(theta, pi)

def log_prob_von_mises_fisher(natural_param, X):
    logcdk = Logcdk.apply
    if len(X.shape) == 1:
        dot = (natural_param * X.to_dense()).sum(dim = -1)
    else:
        dot = sparse_dense_dot(X,natural_param)
    return logcdk(natural_param.shape[-1], natural_param.norm(p=2, dim=-1)) + dot

def log_prob_von_mises_fisher_efficient(pi, kappa, mu, X):
    logcdk = Logcdk.apply
    
    topic_natural_params = kappa.reshape((-1, 1)) * mu

    if pi.shape[-1] == mu.shape[0] + 1:
        topic_natural_params = F.pad(topic_natural_params, [0, 0, 0, 1])
    
    #get norm
    gram_matrix = torch.mm(topic_natural_params, topic_natural_params.T)
    squared_norm = (torch.mm(pi, gram_matrix) * pi).sum(dim=-1)
    norm = squared_norm ** 0.5

    #get dot
    doc_topic_matmul = torch.mm(X, topic_natural_params.T)
    dot = (pi * doc_topic_matmul).sum(dim=-1)
    return logcdk(mu.shape[-1], norm) + dot

def log_prob_von_mises_fisher_single_fixed_kappa_efficient(pi, kappa, mu, X):        
    #get norm
    gram_matrix = torch.mm(mu, mu.T)
    squared_norm = (torch.mm(pi, gram_matrix) * pi).sum(dim=-1)
    norm = squared_norm ** 0.5

    #get dot
    doc_topic_matmul = torch.mm(X, mu.T)
    dot = (pi * doc_topic_matmul).sum(dim=-1)
    return kappa * dot / norm

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

                

class SamJointDistributionWithStickDirUnbiased:
    
    def __init__(self, x, alpha, c0, mu0, kappa1, idx):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa1 = kappa1
        self.idx = idx
        
    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = dist.StickBreakingTransform()(theta)
        pi_chosen = pi[self.idx]
        scaling_factor = theta.shape[0]/self.x.shape[0]
        mu = params['mu']
        return scaling_factor * log_prob_von_mises_fisher_single_fixed_kappa_efficient(pi=pi_chosen, kappa=self.kappa1, mu=mu, X=self.x).sum() \
                + self.c0 * (self.mu0 * mu).sum(dim=-1).sum() \
                + log_prob_stickbreaking_dirichlet(self.alpha, theta, pi).sum()

    
class VptmJointDistributionWithILRDirConjugatePriorUnbiased:

    def __init__(self, x, alpha, c, mu0, v, idx, positive = False):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.idx = idx
        self.positive = positive

    def unnormalized_log_prob(self, params):
      theta = params['theta']
      pi = HelmertILRTransform().inv(theta)
      pi_chosen = pi[self.idx]

      scaling_factor = theta.shape[0]/self.x.shape[0]

      kappa = params['kappa']
      assert kappa.shape == (pi.shape[-1],), f"Expected shape ({pi.shape[-1]},), got {kappa.shape}"

      mu = params['mu']
      if self.positive == True:
          mu = torch.abs(mu)

      return scaling_factor*log_prob_von_mises_fisher_efficient(pi=pi_chosen, kappa=kappa, mu=mu, X=self.x).sum() \
              + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum() \
              + log_prob_ilr_dirichlet(self.alpha, theta, pi).sum()

class VptmJointDistributionWithILRDirLogKappaConjugatePriorUnbiased:

    def __init__(self, x, alpha, c, mu0, v, idx, positive = False):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.idx = idx
        self.positive = positive

    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = HelmertILRTransform().inv(theta)
        pi_chosen = pi[self.idx]

        scaling_factor = theta.shape[0]/self.x.shape[0]
        
        mu = params['mu']
        if self.positive == True:
            mu = torch.abs(mu)

        iota = params['iota']
        assert iota.shape == (mu.shape[0],), f"Expected shape ({mu.shape[0]},), got {iota.shape}"
        kappa = dist.ExpTransform()(iota)
            
        return scaling_factor*log_prob_von_mises_fisher_efficient(pi=pi_chosen, kappa=kappa, mu=mu, X=self.x).sum() \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum() \
                + log_prob_ilr_dirichlet(self.alpha, theta, pi).sum() \
                + dist.ExpTransform().log_abs_det_jacobian(iota, kappa).sum()

class VptmJointDistributionWithILRDirLogKappaMRLWeightedConjugatePriorUnbiased:

    def __init__(self, x, alpha, c, mu0, v, idx, positive = False):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.idx = idx
        self.positive = positive

    def unnormalized_log_prob(self, params):
        theta = params['theta']
        pi = HelmertILRTransform().inv(theta)
        pi_chosen = pi[self.idx]

        scaling_factor = theta.shape[0]/self.x.shape[0]

        iota = params['iota']
        assert iota.shape == (pi.shape[-1],), f"Expected shape ({pi.shape[-1]},), got {iota.shape}"
        kappa = dist.ExpTransform()(iota)
        
        mu = params['mu']
        if self.positive == True:
            mu = torch.abs(mu)

        mrl = ratio(mu.shape[-1]/2, kappa)
            
        return scaling_factor*log_prob_von_mises_fisher_efficient(pi=pi_chosen, kappa=kappa, mu=mu, X=self.x).sum() \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum() \
                + torch.log(mrl).sum() \
                + log_prob_ilr_dirichlet(self.alpha, theta, pi).sum() \
                + dist.ExpTransform().log_abs_det_jacobian(iota, kappa).sum()
                
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


#Transforms

class HelmertILRTransform(dist.transforms.Transform):

    domain = dist.constraints.simplex
    codomain = dist.constraints.real_vector
    bijective = True

    def __eq__(self, other):
        return isinstance(other, HelmertILRTransform)

    @staticmethod
    def _get_helmert_coefs(k, ref):
        n = torch.arange(1, k, dtype=ref.dtype, device=ref.device)
        helmert_coefs = torch.rsqrt(n * (n+1))
        return n, helmert_coefs
    
    def _call(self, x):
        n, helmert_coefs = self._get_helmert_coefs(x.shape[-1], x)
        logx = x.log()
        cumsum_logx = logx[..., :-1].cumsum(-1)
        return (cumsum_logx - n * logx[..., 1:]) * helmert_coefs

    def _inverse(self, y):
        n, helmert_coefs = self._get_helmert_coefs(y.shape[-1]+1, y)
        helmert = y * helmert_coefs
        sum_to_zero = F.pad(helmert.flip(-1).cumsum(-1).flip(-1), [0,1]) - F.pad(n * helmert, [1, 0])
        return torch.softmax(sum_to_zero, dim=-1)

    def log_abs_det_jacobian(self, x, y):
        return - x.log().sum(-1) - 0.5 * math.log(x.shape[-1])

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] - 1,)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)


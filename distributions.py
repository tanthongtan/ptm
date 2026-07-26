import torch
from lcdk import Logcdk, ratio
import torch.nn.functional as F
import torch.distributions as dist
import math

#initialization

class Initialization:

    def __init__(self, num_topic, num_tr, vocab_size, null_component=True):
        self.num_topic = num_topic
        self.num_tr = num_tr
        self.vocab_size = vocab_size
        self.num_components = num_topic + 1 if null_component else num_topic

    def get_initial_mu(self):
        return F.normalize(torch.randn(self.num_topic, self.vocab_size), p=2, dim=-1)

    def get_initial_pi_rn(self, initial_alpha = 10.0):
        init_pi = dist.Dirichlet(torch.full((self.num_components,), initial_alpha)).sample([self.num_tr])
        return HelmertILRTransform()(init_pi)

    def get_initial_rho_rn(self, initial_alpha = 100.0):
        init_rho = dist.Dirichlet(torch.full((self.num_components,), initial_alpha)).sample()
        return HelmertILRTransform()(init_rho)

    def get_initial_a_rn(self, conventional_alpha = 0.5):
        a = conventional_alpha * (self.num_components + torch.randn(()))
        return dist.ExpTransform().inv(a)

    def get_initial_kappa_rn(self, initial_mrl = 0.5):
        random_mrl = torch.randn(self.num_topic) / 100 + initial_mrl
        random_mrl = random_mrl.clip(min=0.01, max=0.99)
        kappa_approx = (random_mrl * (self.vocab_size - random_mrl ** 2)) / (1 - random_mrl ** 2)
        return dist.ExpTransform().inv(kappa_approx)

#probability densities

def log_prob_exponential_log_a(b, a_rn, a):
    return dist.Exponential(b).log_prob(a) + dist.ExpTransform().log_abs_det_jacobian(a_rn, a)

def log_prob_dirichlet_ilr_pi(alpha, pi_rn, pi):
    return dist.Dirichlet(alpha).log_prob(pi) + HelmertILRTransform().inv.log_abs_det_jacobian(pi_rn, pi)

def log_prob_von_mises_fisher_single_datapoint(natural_param, X):
    logcdk = Logcdk.apply
    assert len(X.shape) == 1, "X has to be a single datapoint."
    dot = (natural_param * X.to_dense()).sum(dim = -1)
    return logcdk(natural_param.shape[-1], natural_param.norm(p=2, dim=-1)) + dot

def log_prob_vptm_likelihood(pi, kappa, mu, X):
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

def log_prob_sam_likelihood(pi, kappa, mu, X):        
    #get norm
    gram_matrix = torch.mm(mu, mu.T)
    squared_norm = (torch.mm(pi, gram_matrix) * pi).sum(dim=-1)
    norm = squared_norm ** 0.5

    #get dot
    doc_topic_matmul = torch.mm(X, mu.T)
    dot = (pi * doc_topic_matmul).sum(dim=-1)
    return kappa * dot / norm

def log_prob_bvmfmix_likelihood(mu, kappa, pi, X):
    logcdk = Logcdk.apply

    topic_natural_params = kappa.reshape((-1, 1)) * mu
    dot = torch.mm(X, topic_natural_params.T)
    
    log_norm = logcdk(mu.shape[-1], kappa)
    log_pi = torch.log(pi)
    return torch.logsumexp(log_pi + log_norm + dot, -1)

def log_prob_vmf_conjugate_prior_log_kappa(c, v, mu0, mu, kappa_rn, kappa):
    logcdk = Logcdk.apply
    return v * logcdk(mu0.shape[-1], kappa) + c * kappa * (mu0 * mu).sum(dim=-1) \
        + dist.ExpTransform().log_abs_det_jacobian(kappa_rn, kappa)

                

class SamJointDistributionWithStickDirUnbiased:
    
    def __init__(self, x, alpha, c0, mu0, kappa1, idx):
        self.x = x
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa1 = kappa1
        self.idx = idx
        
    def unnormalized_log_prob(self, params):
        pi_rn = params['pi_rn']
        pi = HelmertILRTransform().inv(pi_rn)
        pi_chosen = pi[self.idx]
        scaling_factor = pi_rn.shape[0]/self.x.shape[0]
        mu = params['mu']
        return scaling_factor * log_prob_sam_likelihood(pi=pi_chosen, kappa=self.kappa1, mu=mu, X=self.x).sum() \
                + self.c0 * (self.mu0 * mu).sum(dim=-1).sum() \
                + log_prob_dirichlet_ilr_pi(self.alpha, pi_rn, pi).sum()

    

class VptmJointDistribution:

    def __init__(self, x, alpha, c, mu0, v, idx, positive = False):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.idx = idx
        self.positive = positive

    def unnormalized_log_prob(self, params):
        pi_rn = params['pi_rn']
        pi = HelmertILRTransform().inv(pi_rn)
        pi_chosen = pi[self.idx]

        scaling_factor = pi_rn.shape[0]/self.x.shape[0]
        
        mu = params['mu']
        if self.positive == True:
            mu = torch.abs(mu)

        kappa_rn = params['kappa_rn']
        assert kappa_rn.shape == (mu.shape[0],), f"Expected shape ({mu.shape[0]},), got {kappa_rn.shape}"
        kappa = dist.ExpTransform()(kappa_rn)
            
        return scaling_factor*log_prob_vptm_likelihood(pi=pi_chosen, kappa=kappa, mu=mu, X=self.x).sum() \
                + log_prob_vmf_conjugate_prior_log_kappa(self.c, self.v, self.mu0, mu, kappa_rn, kappa).sum() \
                + log_prob_dirichlet_ilr_pi(self.alpha, pi_rn, pi).sum() 

class VptmJointDistributionHyperprior:

    def __init__(self, x, alpha, c, mu0, v, b, idx, positive = False):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.idx = idx
        self.b = b
        self.positive = positive

    def unnormalized_log_prob(self, params):
        #dirchlet hyperprior
        rho_rn = params['rho_rn']
        rho = HelmertILRTransform().inv(rho_rn)

        a_rn = params['a_rn']
        assert a_rn.ndim == 0, "a_rn has to be a scalar."
        a = dist.ExpTransform()(a_rn)

        pi_rn = params['pi_rn']
        pi = HelmertILRTransform().inv(pi_rn)
        pi_chosen = pi[self.idx]

        scaling_factor = pi_rn.shape[0]/self.x.shape[0]
        
        mu = params['mu']
        if self.positive == True:
            mu = torch.abs(mu)

        kappa_rn = params['kappa_rn']
        assert kappa_rn.shape == (mu.shape[0],), f"Expected shape ({mu.shape[0]},), got {kappa_rn.shape}"
        kappa = dist.ExpTransform()(kappa_rn)
            
        return scaling_factor*log_prob_vptm_likelihood(pi=pi_chosen, kappa=kappa, mu=mu, X=self.x).sum() \
                + log_prob_vmf_conjugate_prior_log_kappa(self.c, self.v, self.mu0, mu, kappa_rn, kappa).sum() \
                + log_prob_dirichlet_ilr_pi(a * rho, pi_rn, pi).sum() \
                + log_prob_dirichlet_ilr_pi(self.alpha, rho_rn, rho).sum() \
                + log_prob_exponential_log_a(self.b, a_rn, a)
                
class BvmfmixJointDistributionWithStickDirConjugatePrior:
    
    def __init__(self, x, alpha, c, mu0, v, N):
        self.x = x
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.N = N
        
    def unnormalized_log_prob(self, params):
        pi_rn = params['pi_rn']
        pi = HelmertILRTransform().inv(pi_rn)
        assert pi.ndim == 1, "pi has to be a 1D vector"

        scaling_factor = self.N/self.x.shape[0]

        mu = params['mu']

        kappa_rn = params['kappa_rn']
        assert kappa_rn.shape == (mu.shape[0],), f"Expected shape ({mu.shape[0]},), got {kappa_rn.shape}"
        kappa = dist.ExpTransform()(kappa_rn)

        return scaling_factor * log_prob_bvmfmix_likelihood(mu, kappa, pi, self.x).sum() \
                + log_prob_vmf_conjugate_prior_log_kappa(self.c, self.v, self.mu0, mu, kappa_rn, kappa).sum() \
                + log_prob_dirichlet_ilr_pi(self.alpha, pi_rn, pi).sum()


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


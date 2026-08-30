import torch
from lcdk import Logcdk
import torch.nn.functional as F
import torch.distributions as dist
import math

#initialization

class Initialization:

    def __init__(self, M, num_topic, num_tr, vocab_size):
        self.M = M
        self.num_topic = num_topic
        self.num_tr = num_tr
        self.vocab_size = vocab_size

    def get_initial_mu(self):
        return F.normalize(torch.randn(self.M, self.num_topic, self.vocab_size), p=2, dim=-1)

    def get_initial_pi_sphere(self, initial_alpha = 10.0):
        init_pi = dist.Dirichlet(torch.full((self.num_topic,), initial_alpha)).sample([self.M, self.num_tr])
        return init_pi ** 0.5

    def get_initial_kappa(self, initial_mrl = 0.5):
        random_mrl = torch.randn(self.M, self.num_topic) / 100 + initial_mrl
        random_mrl = random_mrl.clip(min=0.01, max=0.99)
        kappa_approx = (random_mrl * (self.vocab_size - random_mrl ** 2)) / (1 - random_mrl ** 2)
        return kappa_approx


#probability densities

def unnormalized_log_prob_spherical_dirichlet(alpha, pi_sphere):
    return ((2*alpha-1)*torch.log(torch.abs(pi_sphere))).sum(dim=-1)

def log_prob_von_mises_fisher_single_datapoint(natural_param, X):
    logcdk = Logcdk.apply
    assert len(X.shape) == 1, "X has to be a single datapoint."
    dot = (natural_param * X.to_dense()).sum(dim = -1)
    return logcdk(natural_param.shape[-1], natural_param.norm(p=2, dim=-1)) + dot

def log_prob_vptm_likelihood(pi, kappa, mu, X):
    logcdk = Logcdk.apply

    M = mu.shape[0]
    vocab_size = mu.shape[-1]
    num_topic = mu.shape[-2]
    S = X.shape[0] // M
    
    topic_natural_params = kappa.unsqueeze(-1) * mu
    
    #get norm
    gram_matrix = torch.matmul(topic_natural_params, topic_natural_params.transpose(-2,-1))
    squared_norm = (torch.matmul(pi, gram_matrix) * pi).sum(dim=-1)
    norm = squared_norm ** 0.5

    #get dot
    doc_topic_matmul = torch.mm(X, topic_natural_params.transpose(-2,-1)
                                .reshape(vocab_size*M, num_topic))
    doc_topic_matmul = doc_topic_matmul.reshape(M, S, num_topic)
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

def log_prob_vmf_conjugate_prior(c, v, mu0, mu, kappa):
    logcdk = Logcdk.apply
    D = mu0.shape[-1]
    return v * logcdk(D, kappa) + c * kappa * (mu0 * mu).sum(dim=-1) + (D-1) * torch.log(kappa) 


class JointDistribution:
       
    def get_unnormalized_log_prob(self, x, idx, beta=1):
        def unnormalized_log_prob(params):
            return self.unnormalized_log_prob_per_chain(params=params, x=x, idx=idx, beta=beta).sum()
        return unnormalized_log_prob
        


class SamJointDistribution:
    
    def __init__(self, alpha, c0, mu0, kappa1):
        self.alpha = alpha
        self.c0 = c0
        self.mu0 = mu0
        self.kappa1 = kappa1
        
    def unnormalized_log_prob_per_chain(self, params, x, idx, beta=1):
        pi_sphere = params['pi_sphere']
        pi = pi_sphere ** 2
        M = pi.shape[0]
        chain_indices = torch.arange(M, dtype=torch.long, device=pi.device).unsqueeze(-1)
        pi_chosen = pi[chain_indices, idx]

        scaling_factor = pi.shape[-2]/pi_chosen.shape[-2]

        mu = params['mu']

        return beta * scaling_factor * log_prob_sam_likelihood(pi=pi_chosen, kappa=self.kappa1, mu=mu, X=x).sum(dim=-1) \
                + self.c0 * (self.mu0 * mu).sum(dim=-1).sum(dim=-1) \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, pi_sphere).sum(dim=-1)



class VptmJointDistribution(JointDistribution):

    def __init__(self, alpha, c, mu0, v, positive = False):        
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.positive = positive

    def unnormalized_log_prob_per_chain(self, params, x, idx, beta=1):
        pi_sphere = params['pi_sphere']
        pi = pi_sphere ** 2
        M = pi.shape[0]
        chain_indices = torch.arange(M, dtype=torch.long, device=pi.device).unsqueeze(-1)
        pi_chosen = pi[chain_indices, idx]

        scaling_factor = pi.shape[-2]/pi_chosen.shape[-2]
        
        mu = params['mu']
        if self.positive == True:
            mu = torch.abs(mu)

        kappa = params['kappa']
        assert kappa.shape == mu.shape[:-1], f"Expected shape {mu.shape[:-1]}, got {kappa.shape}"
            
        return beta * scaling_factor * log_prob_vptm_likelihood(pi=pi_chosen, kappa=kappa, mu=mu, X=x).sum(dim=-1) \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum(dim=-1) \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, pi_sphere).sum(dim=-1) 


                
class BvmfmixJointDistribution:
    
    def __init__(self, alpha, c, mu0, v, N):
        self.alpha = alpha
        self.c = c
        self.mu0 = mu0
        self.v = v
        self.N = N
        
    def unnormalized_log_prob_per_chain(self, params, x, idx, beta=1):
        pi_sphere = params['pi_sphere']
        pi = pi_sphere ** 2
        assert pi.ndim == 2, "pi has to be a 2D array with shape [M, K]"

        scaling_factor = self.N/x.shape[-2]

        mu = params['mu']

        kappa = params['kappa']
        assert kappa.shape == mu.shape[:-1], f"Expected shape {mu.shape[:-1]}, got {kappa.shape}"

        return beta * scaling_factor * log_prob_bvmfmix_likelihood(mu, kappa, pi, x).sum(dim=-1) \
                + log_prob_vmf_conjugate_prior(self.c, self.v, self.mu0, mu, kappa).sum(dim=-1) \
                + unnormalized_log_prob_spherical_dirichlet(self.alpha, pi_sphere).sum(dim=-1)

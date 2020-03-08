import numpy as np
import torch
import torch.distributions as dist

def grad2var(f, independent_axes):
    def result(x, y):
        if independent_axes == None:
            output_shape = torch.tensor(1.)
        elif independent_axes == 1:
            output_shape = torch.ones(x.shape[0])
        x_ = x.detach().requires_grad_(True) 
        y_ = y.detach().requires_grad_(True) 
        f(x_,y_).backward(output_shape)
        return x_.grad, y_.grad
    return result

class SggmcVptm:
    
    def __init__(self, T, gamma, rho, N):
        self.eta = np.sqrt(gamma/N)
        self.c = rho/self.eta
        self.T = T
        
    def transition(self, mu, v_mu, kappa, v_kappa, distribution):
        mu_star = mu.clone()
        v_mu_star = v_mu.clone()
        kappa_star= kappa.clone()
        v_kappa_star = v_kappa.clone()
        for _ in range(self.T):
            #geodesic
            mu_star, v_mu_star = self.spherical_geodesic(mu_star, v_mu_star)
            kappa_star, v_kappa_star = self.positive_geodesic(kappa_star, v_kappa_star)
            #scale
            v_mu_star = torch.exp(-self.c*self.eta/2) * v_mu_star
            v_kappa_star = torch.exp(-self.c*self.eta/2) * v_kappa_star
            #get grads
            grad_mu, grad_kappa = grad2var(distribution, distribution.independent_axes)(mu_star, kappa_star)
            update_mu = self.spherical_projection(mu_star,grad_mu*self.eta+dist.MultivariateNormal(torch.zeros(mu_star.shape[-1]), 2*self.c*self.eta*torch.eye(mu_star.shape[-1])).sample([mu_star.shape[0]]))
            update_kappa = self.positive_projection(kappa_star,grad_kappa*self.eta+dist.MultivariateNormal(torch.zeros(kappa_star.shape[-1]), 2*self.c*self.eta*torch.eye(kappa_star.shape[-1])).sample([kappa_star.shape[0]]))
            #update grads
            v_mu_star= v_mu_star + update_mu
            v_kappa_star = v_kappa_star + update_kappa
            #scale
            v_mu_star = np.exp(-self.c*self.eta/2) * v_mu_star
            v_kappa_star = np.exp(-self.c*self.eta/2) * v_kappa_star
            #geodesic
            mu_star, v_mu_star = self.spherical_geodesic(mu_star, v_mu_star)
            kappa_star, v_kappa_star = self.positive_geodesic(kappa_star, v_kappa_star)
        return mu_star, v_mu_star, kappa_star, v_kappa_star
            
    def spherical_projection(self, x, v):
        v = v - (x*v).sum(dim=-1).unsqueeze(-1) * x
        return v
            
    def spherical_geodesic(self, x, v):
        v_norm = v.norm(p=2, dim = -1).unsqueeze(-1)
        cos_norm_t = torch.cos(v_norm * self.eta/2) 
        sin_norm_t = torch.sin(v_norm * self.eta/2)
        x_new = x * cos_norm_t + v / v_norm * sin_norm_t
        v_new = v * cos_norm_t - v_norm * x * sin_norm_t
        return (x_new, v_new)
    
    def positive_projection(self, x, v):
        return v
    
    def positive_geodesic(self, x, v):
        x_new = x + self.eta/2 * v
        v_new = v
        x_new[x_new<0] = -x_new[x_new<0]
        v_new[x_new<0] = -v_new[x_new<0]
        return (x_new, v_new)
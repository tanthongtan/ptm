import torch
import numpy as np
from torch.autograd import Variable

def ratio(v, z):
    return 0.5*(z/(v-0.5+torch.sqrt((v+0.5)*(v+0.5) + z*z))+z/(v-1+torch.sqrt((v+1)*(v+1) + z*z)))

def logi(v, z):
    return torch.sqrt(z*z + (v+1)*(v+1)) + (v + 0.5)*torch.log(z/(v+0.5+torch.sqrt(z*z+(v+1)*(v+1)))) \
    - 0.5*torch.log(z/2) + (v+0.5)*np.log((2*v+3/2)/(2*(v+1))) - 0.5*np.log(2*np.pi)
    
class Logcdk(torch.autograd.Function):
    """
    Logarithm of the VMF normalization factor
    """
    @staticmethod
    def forward(ctx, d, k):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(k)
        ctx.d = d
        k = k.double()
        answer = (d/2-1)*torch.log(k) - logi(d/2-1, k) - (d/2)*np.log(2*np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors
        d = ctx.d
        k = k.double()
        x = -ratio(d/2, k)
        x = x.float()

        return None,grad_output*Variable(x)
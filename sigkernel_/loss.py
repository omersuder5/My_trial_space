import torch
from traitlets import Any
from .kernels import SignatureKernel, gram
from utils.data import *

def mmd_loss(X: torch.Tensor, Y: torch.Tensor, kernel: Any) -> torch.Tensor:
    # Workaround for sigkernel backward bug when only Y requires grad
    if hasattr(kernel, "compute_Gram") and (Y.requires_grad and not X.requires_grad):
        X = X.detach().requires_grad_(True)

    Kxx = gram(kernel, X, X)
    Kyy = gram(kernel, Y, Y)
    Kxy = gram(kernel, X, Y)

    n = Kxx.shape[0]
    m = Kyy.shape[0]

    sum_xx_off = Kxx.sum() - Kxx.diag().sum()
    sum_yy_off = Kyy.sum() - Kyy.diag().sum()

    return sum_xx_off / (n * (n - 1)) + sum_yy_off / (m * (m - 1)) - 2.0 * Kxy.mean()


def compute_mmd_loss(kernel, X, output, lead_lag = False, lags = None):
    if lead_lag:
        X = batch_lead_lag_transform(X[:,:,1:], X[:,:,0:1], lags) # inputs are (price series, time dimension, lags to use)
        output = batch_lead_lag_transform(output[:,:,1:], output[:,:,0:1], lags)
    return mmd_loss(X, output, kernel)


# def mmd_loss(X: torch.tensor, Y: torch.tensor, kernel: SignatureKernel) -> torch.tensor:
#     '''
#     X: torch.tensor of shape (n_samples, n_features)
#     Y: torch.tensor of shape (n_samples, n_features)
#     kernel: kernel to be used e.g. SignatureKernel
#     '''

#     # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
#     K_XX = gram(kernel, X,X)
#     K_YY = gram(kernel, Y,Y)
#     K_XY = gram(kernel, X,Y)

#     # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
#     n = len(K_XX)
#     m = len(K_YY)

#     mmd = (torch.sum(K_XX[~torch.eye(*K_XX.shape,dtype=torch.bool)]) / (n*(n-1))
#            + torch.sum(K_YY[~torch.eye(*K_YY.shape, dtype=torch.bool)]) / (m*(m-1))
#            - 2*torch.sum(K_XY)/(n*m))

#     return mmd
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

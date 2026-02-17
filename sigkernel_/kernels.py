from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import numpy as np
import torch

# Static kernel class and subclasses
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class Kernel(metaclass=ABCMeta):
    '''
    Base class for static kernels.
    '''

    @abstractmethod
    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.gram_matrix(X, Y)

class LinearKernel(Kernel):

    def __init__(self):
        super().__init__()
        self.static_kernel_type = 'linear'

    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        return matrix_mult(X, Y, transpose_Y=True)

class RBFKernel(Kernel):

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.static_kernel_type = 'rbf'

    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        D2_scaled = squared_euclid_dist(X, Y) / self.sigma**2
        return torch.exp(-D2_scaled)

class RationalQuadraticKernel(Kernel):
    def __init__(self, sigma : float = 1.0, alpha : float = 1.0) -> None:
        super().__init__()
        self.static_kernel_type = 'rq'
        self.alpha = alpha
        self.sigma = sigma

    def gram_matrix(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = squared_euclid_dist(X, Y) / (2 * self.alpha * self.sigma**2)
        return torch.pow((1 + D2_scaled), -self.alpha)

def matrix_mult(X : torch.Tensor, Y : Optional[torch.Tensor] = None, transpose_X : bool = False, transpose_Y : bool = False) -> torch.Tensor:
    subscript_X = '...ji' if transpose_X else '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return torch.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)

def squared_norm(X : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.sum(torch.square(X), dim=dim)

def squared_euclid_dist(X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
    X_n2 = squared_norm(X)
    if Y is None:
        D2 = (X_n2[..., :, None] + X_n2[..., None, :]) - 2 * matrix_mult(X, X, transpose_Y=True)
    else:
        Y_n2 = squared_norm(Y, dim=-1)
        D2 = (X_n2[..., :, None] + Y_n2[..., None, :]) - 2 * matrix_mult(X, Y, transpose_Y=True)
    return D2

# Signature kernel class and functions
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel():
    def __init__(self, n_levels: int = 5, static_kernel: Optional[Kernel] = None) -> None:
        '''
        Parameters
        ----------
        n_levels: int, default=4
            The number of levels of the signature to keep. Higher order terms are truncated
        static_kernel: Kernel, default=None
            The kernel to use for the static kernel. If None, the linear kernel is used.
        '''

        self.n_levels = n_levels
        self.static_kernel = static_kernel if static_kernel is not None else LinearKernel()

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        M = self.static_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1]))).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
        M = torch.diff(torch.diff(M, dim=1), dim=-1) # M[i,j,k,l] = k(X[i,j+1], Y[k,l+1]) - k(X[i,j], Y[k,l+1]) - k(X[i,j+1], Y[k,l]) + k(X[i,j], Y[k,l])
        n_X, n_Y = M.shape[0], M.shape[2]
        K = torch.ones((n_X, n_Y), dtype=M.dtype, device=M.device)
        K += torch.sum(M, dim=(1, -1))
        R = torch.clone(M)
        for _ in range(1, self.n_levels):
            R = M * multi_cumsum(R, axis=(1, -1))
            K += torch.sum(R, dim=(1, -1))

        return K

def multi_cumsum(M: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Computes the exclusive cumulative sum along a given set of axes.

    Args:
        K (torch.Tensor): A matrix over which to compute the cumulative sum
        axis (int or iterable, optional): An axis or a collection of them. Defaults to -1 (the last axis).
    """

    ndim = M.ndim
    axis = [axis] if np.isscalar(axis) else axis
    axis = [ndim+ax if ax < 0 else ax for ax in axis]

    # create slice for exclusive cumsum (slice off last element along given axis then pre-pad with zeros)
    slices = tuple(slice(-1) if ax in axis else slice(None) for ax in range(ndim))
    M = M[slices]

    for ax in axis:
        M = torch.cumsum(M, dim=ax)

    # pre-pad with zeros along the given axis if exclusive cumsum
    pads = tuple(x for ax in reversed(range(ndim)) for x in ((1, 0) if ax in axis else (0, 0)))
    M = torch.nn.functional.pad(M, pads)

    return M


# Kernel call facilitators
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_discretized_signature_kernel(**kwargs):
    static_kernel_type = kwargs['static_kernel_type']
    kernel_sigma = kwargs['kernel_sigma']
    n_levels = kwargs['n_levels']

    if static_kernel_type == 'linear':
        static_kernel = LinearKernel()
    elif static_kernel_type == 'rbf':
        static_kernel = RBFKernel(sigma=kernel_sigma)
    elif static_kernel_type == 'rq':
        static_kernel = RationalQuadraticKernel(sigma=kernel_sigma)
    kernel = SignatureKernel(n_levels=n_levels, static_kernel=static_kernel)

    return kernel

    
def gram(kernel: Any, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Returns Gram matrix K(X,Y) for either:
      - callable kernel: kernel(X,Y)
      - object with compute_Gram: kernel.compute_Gram(X,Y)
    """
    if hasattr(kernel, "compute_Gram"):
        return kernel.compute_Gram(X, Y)
    if hasattr(kernel, "compute_gram"):
        return kernel.compute_gram(X, Y)
    return kernel(X, Y)


# Voltera Kernel implemented in torch for gradient descent -------------------------------------------------------------------------------------------------------
Tensor = torch.Tensor
class VolterraKernel:
    """
    Volterra DP kernel, parameters fitted once from a reference dataset.

    After calling fit(X_ref), the kernel is fixed:
      M := max_{samples,time} ||x_t||_2
      tau := tau_coef / M
      ld  := ld_coef * sqrt(max(0, 1 - (tau*M)^2))
      Gram0 := 1 / (1 - ld^2)

    Kernel between two paths x,y (length T):
      k(x,y) := G[T-1, T-1]
      G[i,j] = 1 + ld^2 * prev / (1 - tau^2 <x_i, y_j>)
      prev = Gram0 if i==0 or j==0 else G[i-1,j-1]
    """

    def __init__(self, *, tau_coef: float = 0.1, ld_coef: float = 0.9, eps: float = 1e-12):
        self.tau_coef = float(tau_coef)
        self.ld_coef = float(ld_coef)
        self.eps = float(eps)

        self._fitted = False
        self.M: Optional[float] = None
        self.tau: Optional[float] = None
        self.ld: Optional[float] = None
        self.Gram0: Optional[float] = None

    def fit(self, X_ref: Tensor) -> "VolterraKernel":
        """
        Fit (tau, ld, Gram0) from reference paths X_ref of shape (N,T,d).
        """
        if X_ref.ndim != 3:
            raise ValueError("X_ref must have shape (N,T,d)")
        # M = max ||x_t||
        norms = torch.linalg.norm(X_ref, dim=-1)  # (N,T)
        M = float(norms.max().clamp_min(self.eps).item())

        tau = float(self.tau_coef / M)
        # ensure ld is real
        ld = float(self.ld_coef * (max(0.0, 1.0 - (tau * M) ** 2) ** 0.5))
        Gram0 = float(1.0 / max(self.eps, 1.0 - ld * ld))

        self.M = M
        self.tau = tau
        self.ld = ld
        self.Gram0 = Gram0
        self._fitted = True
        return self # add this if you want to allow chaining like kernel.fit(X_ref).compute_Gram(X,Y)

    def __call__(self, X: Tensor, Y: Tensor) -> Tensor:
        return self.compute_Gram(X, Y)

    def compute_Gram(self, X: Tensor, Y: Tensor) -> Tensor:
        if not self._fitted:
            raise RuntimeError("VolterraKernel must be fitted first: call kernel.fit(X_ref).")

        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must have shape (n,T,d) and (m,T,d)")
        n, T, d = X.shape
        m, T2, d2 = Y.shape
        if T != T2:
            raise ValueError(f"T mismatch: {T} vs {T2}")
        if d != d2:
            raise ValueError(f"d mismatch: {d} vs {d2}")

        device, dtype = X.device, X.dtype

        tau2 = torch.tensor(self.tau * self.tau, device=device, dtype=dtype)
        ld2 = torch.tensor(self.ld * self.ld, device=device, dtype=dtype)
        Gram0 = torch.tensor(self.Gram0, device=device, dtype=dtype)
        eps = torch.tensor(self.eps, device=device, dtype=dtype)

        K = torch.empty((n, m), device=device, dtype=dtype)

        # vectorized over Y-batch (m) per fixed X[a]
        for a in range(n):
            G_prev = torch.empty((m, T), device=device, dtype=dtype)

            for i in range(T):
                xi = X[a, i, :]  # (d,)
                dots = torch.einsum("mtd,d->mt", Y, xi)  # (m,T)
                denom = torch.clamp(1.0 - tau2 * dots, min=eps)

                if i == 0:
                    prev_diag = Gram0.expand(m, T)
                else:
                    prev_diag = torch.cat([Gram0.expand(m, 1), G_prev[:, :-1]], dim=1)

                G_prev = 1.0 + ld2 * prev_diag / denom

            # terminal value per Y[b]
            K[a, :] = G_prev[:, -1]

        return K

    def spec(self) -> dict:
        """
        Safe-to-save kernel spec (no objects).
        """
        if not self._fitted:
            return {
                "kernel_mode": "sequential",
                "kernel_name": "VolterraKernel",
                "kernel_params": {"tau_coef": self.tau_coef, "ld_coef": self.ld_coef, "eps": self.eps},
                "kernel_str": f"VolterraKernel(unfitted, tau_coef={self.tau_coef}, ld_coef={self.ld_coef})",
            }
        return {
            "kernel_mode": "sequential",
            "kernel_name": "VolterraKernel",
            "kernel_params": {
                "tau_coef": self.tau_coef,
                "ld_coef": self.ld_coef,
                "eps": self.eps,
                "M": self.M,
                "tau": self.tau,
                "ld": self.ld,
                "Gram0": self.Gram0,
            },
            "kernel_str": f"VolterraKernel(tau={self.tau:.3g}, ld={self.ld:.3g}, M={self.M:.3g})",
        }

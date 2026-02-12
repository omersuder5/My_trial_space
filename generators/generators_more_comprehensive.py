from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal

import math
import torch
import torch.nn as nn


Tensor = torch.Tensor


# ----------------------------
# Noise specification
# ----------------------------

@dataclass(frozen=True)
class NoiseSpec:
    """
    Generic IID noise generator.

    dist:
      - "normal":   Normal(loc, scale)
      - "t":        StudentT(df) with location/scale applied
      - "gamma":    Gamma(concentration, rate)   (rate = 1/scale)
      - "laplace":  Laplace(loc, scale)
      - "uniform":  Uniform(low, high)

    params: dictionary of distribution params. See _sample_iid.
    """
    dist: Literal["normal", "t", "gamma", "laplace", "uniform"] = "normal"
    params: dict = None

    def __post_init__(self):
        object.__setattr__(self, "params", {} if self.params is None else dict(self.params))

    def sample(self, shape: Tuple[int, ...], device=None, dtype=None) -> Tensor:
        return _sample_iid(self.dist, self.params, shape, device=device, dtype=dtype)


def _sample_iid(
    dist: str,
    params: dict,
    shape: Tuple[int, ...],
    device=None,
    dtype=None,
) -> Tensor:
    device = device or torch.device("cpu")
    dtype = dtype or torch.get_default_dtype()
    params = {} if params is None else params

    if dist == "normal":
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        return loc + scale * torch.randn(*shape, device=device, dtype=dtype)

    if dist == "t":
        df = float(params.get("df", 5.0))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        # torch StudentT is in torch.distributions
        td = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        return td.sample(shape).to(device=device, dtype=dtype)

    if dist == "gamma":
        concentration = float(params.get("concentration", 2.0))
        rate = float(params.get("rate", 1.0))  # rate = 1/scale
        gd = torch.distributions.Gamma(concentration=concentration, rate=rate)
        return gd.sample(shape).to(device=device, dtype=dtype)

    if dist == "laplace":
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        ld = torch.distributions.Laplace(loc=loc, scale=scale)
        return ld.sample(shape).to(device=device, dtype=dtype)

    if dist == "uniform":
        low = float(params.get("low", -1.0))
        high = float(params.get("high", 1.0))
        ud = torch.distributions.Uniform(low=low, high=high)
        return ud.sample(shape).to(device=device, dtype=dtype)

    raise ValueError(f"Unknown dist='{dist}'")


# ----------------------------
# Base generator
# ----------------------------

class ProcGenBase(nn.Module):
    """
    Base class for simple stochastic process generators.

    Produces sequences of shape (N, T, d), where:
      N = number of independent sequences
      T = seq_len
      d = seq_dim

    Subclasses implement:
      - _simulate_innovations(...) or directly simulate x.

    Notes:
      - This is not meant for training, but uses nn.Module for uniformity,
        state_dict support, device/dtype propagation, etc.
      - Noise is supplied as a NoiseSpec (IID) or explicit tensor eps.
    """
    def __init__(
        self,
        seq_len: int,
        seq_dim: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_dim <= 0:
            raise ValueError("seq_dim must be positive")

        self.seq_len = int(seq_len)
        self.seq_dim = int(seq_dim)

        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()

        self.register_buffer("_dummy", torch.empty((), device=device, dtype=dtype), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def _ensure_eps(
        self,
        N: int,
        eps: Optional[Tensor],
        noise: Optional[NoiseSpec],
    ) -> Tensor:
        if eps is not None:
            if eps.ndim != 3:
                raise ValueError("eps must have shape (N, T, d)")
            if eps.shape[0] != N or eps.shape[1] != self.seq_len or eps.shape[2] != self.seq_dim:
                raise ValueError(f"eps must have shape (N={N}, T={self.seq_len}, d={self.seq_dim})")
            return eps.to(device=self.device, dtype=self.dtype)

        if noise is None:
            noise = NoiseSpec("normal", {"loc": 0.0, "scale": 1.0})

        return noise.sample((N, self.seq_len, self.seq_dim), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(
        self,
        N: int = 1,
        *,
        eps: Optional[Tensor] = None,
        noise: Optional[NoiseSpec] = None,
        x0: Union[float, Tensor] = 0.0,
    ) -> Tensor:
        """
        Generate N independent sequences.

        Args:
          N: number of sequences
          eps: optional explicit innovations/noise, shape (N,T,d)
          noise: optional NoiseSpec if eps is not provided
          x0: scalar or tensor initial value; broadcastable to (N,d)

        Returns:
          x: shape (N,T,d)
        """
        if N <= 0:
            raise ValueError("N must be positive")
        return self._generate_impl(N=N, eps=eps, noise=noise, x0=x0)

    def _generate_impl(self, N: int, eps: Optional[Tensor], noise: Optional[NoiseSpec], x0) -> Tensor:
        raise NotImplementedError


# ----------------------------
# ARMA (includes AR, MA edge cases)
# ----------------------------

class ARMAGen(ProcGenBase):
    """
    ARMA(p,q) with optional mean (drift) and scale on innovations.

    x_t = mu + sum_{i=1}^p phi_i x_{t-i} + eps_t + sum_{j=1}^q theta_j eps_{t-j}

    - If p=0 -> pure MA(q)
    - If q=0 -> pure AR(p)
    - If p=q=0 -> white noise with mean mu

    Conventions:
      - phi: (..., p) or (p,) applied per-dimension if you pass (d,p)
      - theta: (..., q)
      - mu: scalar or (d,)
    """
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        p: int,
        q: int,
        *,
        phi: Optional[Tensor] = None,
        theta: Optional[Tensor] = None,
        mu: Union[float, Tensor] = 0.0,
    ):
        super().__init__(seq_len=seq_len, seq_dim=seq_dim)
        if p < 0 or q < 0:
            raise ValueError("p,q must be >= 0")
        self.p = int(p)
        self.q = int(q)

        # Store parameters as buffers (not trainable by default; change to nn.Parameter if you want)
        if self.p > 0:
            if phi is None:
                phi = torch.zeros(seq_dim, self.p, device=self.device, dtype=self.dtype)
            phi = torch.as_tensor(phi, device=self.device, dtype=self.dtype)
            if phi.ndim == 1:
                phi = phi.unsqueeze(0).repeat(seq_dim, 1)  # (d,p)
            if phi.shape != (seq_dim, self.p):
                raise ValueError(f"phi must have shape (d={seq_dim}, p={self.p}) or (p,)")
            self.register_buffer("phi", phi)
        else:
            self.register_buffer("phi", torch.empty(seq_dim, 0, device=self.device, dtype=self.dtype))

        if self.q > 0:
            if theta is None:
                theta = torch.zeros(seq_dim, self.q, device=self.device, dtype=self.dtype)
            theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype)
            if theta.ndim == 1:
                theta = theta.unsqueeze(0).repeat(seq_dim, 1)  # (d,q)
            if theta.shape != (seq_dim, self.q):
                raise ValueError(f"theta must have shape (d={seq_dim}, q={self.q}) or (q,)")
            self.register_buffer("theta", theta)
        else:
            self.register_buffer("theta", torch.empty(seq_dim, 0, device=self.device, dtype=self.dtype))

        mu_t = torch.as_tensor(mu, device=self.device, dtype=self.dtype)
        if mu_t.ndim == 0:
            mu_t = mu_t.repeat(seq_dim)  # (d,)
        if mu_t.shape != (seq_dim,):
            raise ValueError(f"mu must be scalar or shape (d={seq_dim},)")
        self.register_buffer("mu", mu_t)

    def _generate_impl(self, N: int, eps: Optional[Tensor], noise: Optional[NoiseSpec], x0) -> Tensor:
        eps = self._ensure_eps(N, eps, noise)  # (N,T,d)

        x0_t = torch.as_tensor(x0, device=self.device, dtype=self.dtype)
        if x0_t.ndim == 0:
            x_prev = x0_t.view(1, 1).repeat(N, self.seq_dim)  # (N,d)
        else:
            x_prev = x0_t
            if x_prev.shape != (N, self.seq_dim):
                # allow (d,) broadcast to (N,d)
                if x_prev.shape == (self.seq_dim,):
                    x_prev = x_prev.view(1, self.seq_dim).repeat(N, 1)
                else:
                    raise ValueError("x0 must be scalar, (d,), or (N,d)")

        T = self.seq_len
        d = self.seq_dim
        x = torch.zeros((N, T, d), device=self.device, dtype=self.dtype)

        # Maintain short histories for x and eps
        x_hist = [x_prev]  # x_{t-1}, x_{t-2}, ...
        eps_hist = [torch.zeros((N, d), device=self.device, dtype=self.dtype) for _ in range(self.q)]

        for t in range(T):
            e_t = eps[:, t, :]  # (N,d)

            ar_term = 0.0
            if self.p > 0:
                # x_hist[0] = x_{t-1}, ...
                # ensure we have p elements
                while len(x_hist) < self.p:
                    x_hist.append(torch.zeros((N, d), device=self.device, dtype=self.dtype))
                # sum_i phi_i x_{t-i}
                # phi: (d,p)
                # stack x_{t-1..t-p}: (N,d,p)
                x_stack = torch.stack(x_hist[: self.p], dim=-1)
                ar_term = (x_stack * self.phi.unsqueeze(0)).sum(dim=-1)  # (N,d)

            ma_term = 0.0
            if self.q > 0:
                # eps_hist[0] = eps_{t-1}, ...
                # stack eps_{t-1..t-q}: (N,d,q)
                e_stack = torch.stack(eps_hist[: self.q], dim=-1)
                ma_term = (e_stack * self.theta.unsqueeze(0)).sum(dim=-1)  # (N,d)

            x_t = self.mu.view(1, d) + ar_term + e_t + ma_term
            x[:, t, :] = x_t

            # update histories
            x_hist.insert(0, x_t)
            if len(x_hist) > max(self.p, 1):
                x_hist = x_hist[: max(self.p, 1)]

            if self.q > 0:
                eps_hist.insert(0, e_t)
                eps_hist = eps_hist[: self.q]

        return x


# ----------------------------
# GARCH (univariate per-dimension, optionally independent dims)
# ----------------------------

class GARCHGen(ProcGenBase):
    """
    Diagonal GARCH(p,q) per dimension (independent across dimensions).

    Let z_t be IID noise (user-chosen). Innovations:
      eps_t = sigma_t * z_t

    Variance recursion (per dimension):
      sigma_t^2 = omega + sum_{i=1}^p alpha_i * eps_{t-i}^2 + sum_{j=1}^q beta_j * sigma_{t-j}^2

    Output series:
      x_t = mu + eps_t
    If you want returns only, set mu=0.

    Notes:
      - "diagonal" means each dimension uses its own parameters and history, no cross terms.
      - Works with non-Gaussian z_t via NoiseSpec or explicit z.
    """
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        p: int,
        q: int,
        *,
        omega: Union[float, Tensor] = 0.1,
        alpha: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
        mu: Union[float, Tensor] = 0.0,
        # optional initial variance
        sigma2_0: Union[float, Tensor] = 1.0,
        clamp_sigma2_min: float = 1e-12,
    ):
        super().__init__(seq_len=seq_len, seq_dim=seq_dim)
        if p < 0 or q < 0:
            raise ValueError("p,q must be >= 0")
        if p == 0 and q == 0:
            raise ValueError("GARCH with p=q=0 is degenerate; use white noise (ARMA(0,0)) instead.")

        self.p = int(p)
        self.q = int(q)
        self.clamp_sigma2_min = float(clamp_sigma2_min)

        omega_t = torch.as_tensor(omega, device=self.device, dtype=self.dtype)
        if omega_t.ndim == 0:
            omega_t = omega_t.repeat(seq_dim)
        if omega_t.shape != (seq_dim,):
            raise ValueError(f"omega must be scalar or shape (d={seq_dim},)")
        self.register_buffer("omega", omega_t)

        if self.p > 0:
            if alpha is None:
                alpha = torch.zeros(seq_dim, self.p, device=self.device, dtype=self.dtype)
            alpha = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0).repeat(seq_dim, 1)
            if alpha.shape != (seq_dim, self.p):
                raise ValueError(f"alpha must have shape (d={seq_dim}, p={self.p}) or (p,)")
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer("alpha", torch.empty(seq_dim, 0, device=self.device, dtype=self.dtype))

        if self.q > 0:
            if beta is None:
                beta = torch.zeros(seq_dim, self.q, device=self.device, dtype=self.dtype)
            beta = torch.as_tensor(beta, device=self.device, dtype=self.dtype)
            if beta.ndim == 1:
                beta = beta.unsqueeze(0).repeat(seq_dim, 1)
            if beta.shape != (seq_dim, self.q):
                raise ValueError(f"beta must have shape (d={seq_dim}, q={self.q}) or (q,)")
            self.register_buffer("beta", beta)
        else:
            self.register_buffer("beta", torch.empty(seq_dim, 0, device=self.device, dtype=self.dtype))

        mu_t = torch.as_tensor(mu, device=self.device, dtype=self.dtype)
        if mu_t.ndim == 0:
            mu_t = mu_t.repeat(seq_dim)
        if mu_t.shape != (seq_dim,):
            raise ValueError(f"mu must be scalar or shape (d={seq_dim},)")
        self.register_buffer("mu", mu_t)

        sigma2_0_t = torch.as_tensor(sigma2_0, device=self.device, dtype=self.dtype)
        if sigma2_0_t.ndim == 0:
            sigma2_0_t = sigma2_0_t.repeat(seq_dim)
        if sigma2_0_t.shape != (seq_dim,):
            raise ValueError(f"sigma2_0 must be scalar or shape (d={seq_dim},)")
        self.register_buffer("sigma2_0", sigma2_0_t)

    def _generate_impl(self, N: int, eps: Optional[Tensor], noise: Optional[NoiseSpec], x0) -> Tensor:
        # Here eps argument is interpreted as z (standardized shocks), not innovations.
        z = self._ensure_eps(N, eps, noise)  # (N,T,d), IID

        T = self.seq_len
        d = self.seq_dim

        # histories
        eps2_hist = [torch.zeros((N, d), device=self.device, dtype=self.dtype) for _ in range(self.p)]
        sig2_hist = [self.sigma2_0.view(1, d).repeat(N, 1)]

        x = torch.zeros((N, T, d), device=self.device, dtype=self.dtype)

        for t in range(T):
            sig2_t = self.omega.view(1, d)

            if self.p > 0:
                e2_stack = torch.stack(eps2_hist[: self.p], dim=-1)  # (N,d,p)
                sig2_t = sig2_t + (e2_stack * self.alpha.unsqueeze(0)).sum(dim=-1)

            if self.q > 0:
                s2_stack = torch.stack(sig2_hist[: self.q], dim=-1)  # (N,d,q)
                sig2_t = sig2_t + (s2_stack * self.beta.unsqueeze(0)).sum(dim=-1)

            sig2_t = torch.clamp(sig2_t, min=self.clamp_sigma2_min)
            sigma_t = torch.sqrt(sig2_t)

            eps_t = sigma_t * z[:, t, :]
            x_t = self.mu.view(1, d) + eps_t
            x[:, t, :] = x_t

            # update histories
            if self.p > 0:
                eps2_hist.insert(0, eps_t.square())
                eps2_hist = eps2_hist[: self.p]
            sig2_hist.insert(0, sig2_t)
            sig2_hist = sig2_hist[: max(self.q, 1)]

        return x


# ----------------------------
# Minimal usage examples
# ----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Generate 8 independent AR(2) sequences in 1D
    ar2 = ARMAGen(seq_len=200, seq_dim=1, p=2, q=0, phi=torch.tensor([0.7, -0.2]), mu=0.0)
    x_ar = ar2.generate(N=8, noise=NoiseSpec("normal", {"loc": 0.0, "scale": 1.0}))

    # Pure MA(3) with t noise
    ma3 = ARMAGen(seq_len=200, seq_dim=1, p=0, q=3, theta=torch.tensor([0.5, 0.3, -0.1]))
    x_ma = ma3.generate(N=4, noise=NoiseSpec("t", {"df": 7, "loc": 0.0, "scale": 1.0}))

    # ARMA(1,1) with explicit eps tensor (overrides noise spec)
    arma11 = ARMAGen(seq_len=100, seq_dim=2, p=1, q=1,
                     phi=torch.tensor([[0.5], [0.2]]), theta=torch.tensor([[0.4], [0.1]]))
    eps = torch.randn(3, 100, 2)
    x_arma = arma11.generate(N=3, eps=eps)

    # GARCH(1,1) with gamma shocks (non-symmetric) in 1D
    g11 = GARCHGen(seq_len=500, seq_dim=1, p=1, q=1,
                   omega=0.05, alpha=torch.tensor([0.08]), beta=torch.tensor([0.90]), mu=0.0)
    z = g11.generate(N=2, noise=NoiseSpec("gamma", {"concentration": 2.0, "rate": 1.0}))  # NOTE: here z are returns, see below
    # If you want z_t as standardized gamma, provide eps=... after standardizing yourself.

    print(x_ar.shape, x_ma.shape, x_arma.shape)

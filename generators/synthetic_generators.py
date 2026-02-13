from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict
import torch
import torch.nn as nn

Tensor = torch.Tensor


# -------------------------
# noise
# -------------------------

@dataclass
class Noise:
    kind: Literal["normal", "t", "gamma"] = "normal"
    params: Dict = None

    def sample(self, shape, device=None, dtype=None) -> Tensor:
        params = {} if self.params is None else self.params
        device = device or torch.device("cpu")
        dtype = dtype or torch.get_default_dtype()

        if self.kind == "normal":
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return loc + scale * torch.randn(*shape, device=device, dtype=dtype)

        if self.kind == "t":
            df = float(params.get("df", 5.0))
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return torch.distributions.StudentT(df, loc=loc, scale=scale).sample(shape).to(device=device, dtype=dtype)

        if self.kind == "gamma":
            # Gamma(concentration, rate). Default: mean=2 if conc=2, rate=1
            conc = float(params.get("concentration", 2.0))
            rate = float(params.get("rate", 1.0))
            return torch.distributions.Gamma(conc, rate=rate).sample(shape).to(device=device, dtype=dtype)

        raise ValueError(f"unknown noise kind: {self.kind}")


# -------------------------
# base
# -------------------------

class Proc(nn.Module):
    def __init__(self, T: int, d: int = 1):
        super().__init__()
        self.T, self.d = int(T), int(d)
        self.register_buffer("_dummy", torch.empty(()), persistent=False)

    @property
    def device(self): return self._dummy.device
    @property
    def dtype(self): return self._dummy.dtype

    @torch.no_grad()
    def generate(self, N: int = 1, noise: Optional[Noise] = None, eps: Optional[Tensor] = None) -> Tensor:
        # returns (N,T,d)
        if eps is None:
            noise = noise or Noise("normal")
            eps = noise.sample((N, self.T, self.d), device=self.device, dtype=self.dtype)
        else:
            eps = eps.to(self.device, self.dtype)
        return self._gen(N, eps)

    def _gen(self, N: int, eps: Tensor) -> Tensor:
        raise NotImplementedError


# -------------------------
# ARMA (AR if q=0, MA if p=0)
# -------------------------

class ARMA(Proc):
    # x_t = sum_i phi_i x_{t-i} + eps_t + sum_j theta_j eps_{t-j}
    def __init__(self, T: int, p: int, q: int, phi=None, theta=None, d: int = 1):
        super().__init__(T, d)
        self.p, self.q = int(p), int(q)

        if self.p:
            phi = torch.zeros(self.p) if phi is None else torch.as_tensor(phi)
            if phi.numel() != self.p: raise ValueError("phi size mismatch")
            self.register_buffer("phi", phi.to(self.dtype))
        else:
            self.register_buffer("phi", torch.empty(0, dtype=self.dtype))

        if self.q:
            theta = torch.zeros(self.q) if theta is None else torch.as_tensor(theta)
            if theta.numel() != self.q: raise ValueError("theta size mismatch")
            self.register_buffer("theta", theta.to(self.dtype))
        else:
            self.register_buffer("theta", torch.empty(0, dtype=self.dtype))

    def _gen(self, N: int, eps: Tensor) -> Tensor:
        T, d = self.T, self.d
        x = torch.zeros((N, T, d), device=self.device, dtype=self.dtype)

        for t in range(T):
            ar = 0.0
            if self.p:
                for i in range(1, self.p + 1):
                    if t - i >= 0:
                        ar = ar + self.phi[i - 1] * x[:, t - i, :]

            ma = 0.0
            if self.q:
                for j in range(1, self.q + 1):
                    if t - j >= 0:
                        ma = ma + self.theta[j - 1] * eps[:, t - j, :]

            x[:, t, :] = ar + eps[:, t, :] + ma

        return x


# -------------------------
# GARCH(1,1) only, simplest useful one
# -------------------------

class GARCH11(Proc):
    # sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
    # x_t = sigma_t * z_t    where z_t is the provided noise eps
    def __init__(self, T: int, omega: float, alpha: float, beta: float, d: int = 1, sigma2_0: float = 1.0):
        super().__init__(T, d)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma2_0 = float(sigma2_0)

    def _gen(self, N: int, z: Tensor) -> Tensor:
        T, d = self.T, self.d
        x = torch.zeros((N, T, d), device=self.device, dtype=self.dtype)

        sigma2 = torch.full((N, d), self.sigma2_0, device=self.device, dtype=self.dtype)
        eps_prev = torch.zeros((N, d), device=self.device, dtype=self.dtype)

        for t in range(T):
            sigma2 = self.omega + self.alpha * (eps_prev ** 2) + self.beta * sigma2
            sigma2 = torch.clamp(sigma2, min=1e-12)
            eps_t = torch.sqrt(sigma2) * z[:, t, :]
            x[:, t, :] = eps_t
            eps_prev = eps_t

        return x
    
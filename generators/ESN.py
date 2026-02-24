from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Union, Literal, Dict

import torch
import torch.nn as nn

Tensor = torch.Tensor

def _get_activation(name_or_fn: Union[str, Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    if callable(name_or_fn):
        return name_or_fn
    name = str(name_or_fn).lower()
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unknown activation: {name_or_fn}")

def _ma_filter(e: Tensor, theta: Tensor) -> Tensor:
    """
    e: (N,T,m) iid innovations
    theta: (q,) MA coefficients
    y_t = e_t + sum_{j=1}^q theta[j-1] e_{t-j}
    """
    theta = theta.reshape(-1)
    q = int(theta.numel())
    if q == 0:
        return e

    N, T, m = e.shape
    y = e.clone()
    for j in range(1, q + 1):
        y[:, j:, :] = y[:, j:, :] + theta[j - 1] * e[:, : T - j, :]
    return y


def rescale_spectral_radius(A: Tensor, target_rho: float) -> Tensor:
    """
    Rescale A so that spectral radius max |lambda_i| equals target_rho.
    One-time constraint at init, A remains fixed afterwards.
    """
    if not (0.0 < target_rho < 1.0):
        raise ValueError("target_rho must be in (0,1)")
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(A)
        rho = eigvals.abs().max().real
        if rho <= 0:
            return A
        return A * (target_rho / rho)


class ESNGenerator(nn.Module):
    """
    ESN generator (A, C fixed; only W trainable):

      X_t = sigma( A X_{t-1} + C xi_t )
      Z_t = W X_t + eta_t + t_tilt

    xi_t ~ IID Normal(0, xi_scale^2 I_m)
    eta_t ~ IID Normal(0, eta_scale^2 I_d)

    Shapes:
      A: (h, h), spectral radius constrained to target_rho < 1 at init
      C: (h, m)
      W: (d, h)  (trainable)
      X_t: (N, h)
      Z_t: (N, d)

    forward returns Z of shape (N, T, d).
    """

    def __init__(
        self,
        A: Tensor,
        C: Tensor,
        out_dim: int,
        *,
        activation: Union[str, Callable[[Tensor], Tensor]] = "tanh",
        xi_scale: float = 1.0,
        eta_scale: float = 1.0,
        target_rho: float = 0.9,
        xi_ma_theta: Optional[Tensor] = None,
        t_tilt: Optional[Tensor] = None,
        W_init_std: float = 0.1,
    ):
        super().__init__()
        A = torch.as_tensor(A)
        C = torch.as_tensor(C)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square (h,h)")
        if C.ndim != 2 or C.shape[0] != A.shape[0]:
            raise ValueError("C must have shape (h,m) with same h as A")

        self.h = int(A.shape[0])
        self.m = int(C.shape[1])
        self.d = int(out_dim)

        A = rescale_spectral_radius(A, float(target_rho))
        self.register_buffer("A", A)
        self.register_buffer("C", C)

        W0 = torch.randn(self.d, self.h, device=A.device, dtype=A.dtype) * float(W_init_std)
        self.W = nn.Parameter(W0)

        self.activation = _get_activation(activation)
        self.activation_name = activation if isinstance(activation, str) else getattr(activation, "__name__", "custom")
        self.xi_scale = float(xi_scale)
        self.eta_scale = float(eta_scale)

        if t_tilt is None:
            self.register_buffer("t_tilt", None, persistent=False)
        else:
            self.register_buffer("t_tilt", torch.as_tensor(t_tilt, device=A.device, dtype=A.dtype), persistent=False)

        if xi_ma_theta is None:
            self.register_buffer("xi_ma_theta", None, persistent=False)
        else:
            self.register_buffer("xi_ma_theta", torch.as_tensor(xi_ma_theta, device=A.device, dtype=A.dtype), persistent=False)
        if self.xi_ma_theta is not None and self.xi_ma_theta.ndim != 1:
            raise ValueError("xi_ma_theta must be 1D (q,).")

    @torch.no_grad()
    def sample_noise(self, N: int, T: int) -> tuple[Tensor, Tensor]:
        device, dtype = self.A.device, self.A.dtype
        xi_ma_theta = self.xi_ma_theta

        # iid innovations
        xi_e = torch.randn(N, T, self.m, device=device, dtype=dtype) * self.xi_scale
        eta = torch.randn(N, T, self.d, device=device, dtype=dtype) * self.eta_scale

        # colored input noise
        if xi_ma_theta is not None:
            th = torch.as_tensor(xi_ma_theta, device=device, dtype=dtype)
            xi = _ma_filter(xi_e, th)
        else:
            xi = xi_e

        return xi, eta
    
    def forward(
        self,
        T: int,
        *,
        N: int = 1,
        x0: Optional[Tensor] = None,     # (N,h) or (h,) or None
        xi: Optional[Tensor] = None,     # (N,T,m) or None
        eta: Optional[Tensor] = None,    # (N,T,d) or None
        return_states: bool = False,
    ):
        device, dtype = self.A.device, self.A.dtype

        if x0 is None:
            x = torch.zeros(N, self.h, device=device, dtype=dtype)
        else:
            x0 = torch.as_tensor(x0, device=device, dtype=dtype)
            if x0.shape == (self.h,):
                x = x0.view(1, self.h).repeat(N, 1)
            elif x0.shape == (N, self.h):
                x = x0
            else:
                raise ValueError("x0 must be (h,) or (N,h)")

        if xi is None or eta is None:
            xi_s, eta_s = self.sample_noise(N, T)
            if xi is None:
                xi = xi_s
            if eta is None:
                eta = eta_s
        else:
            xi = torch.as_tensor(xi, device=device, dtype=dtype)
            eta = torch.as_tensor(eta, device=device, dtype=dtype)

        if xi.shape != (N, T, self.m):
            raise ValueError(f"xi must have shape (N,T,m)=({N},{T},{self.m})")
        if eta.shape != (N, T, self.d):
            raise ValueError(f"eta must have shape (N,T,d)=({N},{T},{self.d})")

        if self.t_tilt is None:
            tilt = None
        else:
            tt = self.t_tilt
            if tt.ndim == 1 and tt.shape == (self.d,):
                tilt = tt.view(1, 1, self.d)
            elif tt.ndim == 2 and tt.shape == (T, self.d):
                tilt = tt.view(1, T, self.d)
            elif tt.ndim == 3 and tt.shape[1:] == (T, self.d):
                tilt = tt
            else:
                raise ValueError("t_tilt must be (d,) or (T,d) or (1,T,d) or broadcastable to (N,T,d)")

        Z = torch.empty(N, T, self.d, device=device, dtype=dtype)
        Xhist = torch.empty(N, T, self.h, device=device, dtype=dtype) if return_states else None

        A, C, W = self.A, self.C, self.W
        act = self.activation

        for t in range(T):
            x = act(x @ A.T + xi[:, t, :] @ C.T)
            z = x @ W.T + eta[:, t, :]
            if tilt is not None:
                z = z + tilt[:, t, :]
            Z[:, t, :] = z
            if return_states:
                Xhist[:, t, :] = x

        return (Z, Xhist) if return_states else Z

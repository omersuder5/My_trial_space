from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Literal, Dict, List, Deque, Tuple
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from sigkernel_.loss import compute_mmd_loss

Tensor = torch.Tensor


@torch.no_grad()
def _as_float64(x: Tensor) -> Tensor:
    return x.to(dtype=torch.float64)


@torch.no_grad()
def _sample_states_once(
    esn,
    *,
    T: int,
    N_model: int,
    dtype: torch.dtype,
    device: torch.device,
    xi: Optional[Tensor] = None,
    eta: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      Z0: (N_model, T, d)  (ignored for optimization, but sometimes useful)
      Xhist: (N_model, T, h)  reservoir states, fixed during optimization
    """
    esn = esn.to(device=device, dtype=dtype).eval()

    if eta is None:
        eta = torch.zeros(N_model, T, esn.d, device=device, dtype=dtype)

    out = esn(T=T, N=N_model, xi=xi, eta=eta, return_states=True)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise ValueError("esn(T=..., N=..., return_states=True) must return (Z, Xhist)")
    Z0, Xhist = out

    if Z0.shape[:2] != (N_model, T):
        raise ValueError(f"Z0 shape {tuple(Z0.shape)} incompatible with (N_model,T)=({N_model},{T})")
    if Xhist.shape[:2] != (N_model, T):
        raise ValueError(f"Xhist shape {tuple(Xhist.shape)} incompatible with (N_model,T)=({N_model},{T})")

    return Z0, Xhist


def fit_W_by_lbfgs_mmd(
    *,
    esn,
    Z_target: Tensor,                              # (N_target, T, d), fixed
    kernel: Any,
    kernel_mode: Literal["static", "sequential"],
    N_model: int,                                  # number of ESN paths used inside MMD
    lead_lag: bool = False,
    lags: int = 1,
    max_iter: int = 200,
    lr: float = 1.0,
    history_size: int = 20,
    tol_grad: float = 1e-12,
    tol_change: float = 1e-12,
    force_float64: bool = True,
    xi: Optional[Tensor] = None,                    # optional fixed drive for model sampling (N_model,T,m)
    eta: Optional[Tensor] = None,                   # optional fixed output noise (N_model,T,d); default 0
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optimizes W only:
        min_W MMD( Z_target , Z_model(W) )
    where Z_model(W) = Xhist @ W^T (+ tilt if your esn includes it in forward; here we ignore tilt).

    Notes:
    - Z_target is fixed (no resampling).
    - Model states Xhist are sampled once (using xi, eta), then held fixed.
    - If kernel is sigkernel-based, force_float64 should stay True.
    """
    if Z_target.ndim != 3:
        raise ValueError("Z_target must have shape (N_target,T,d)")
    N_target, T, d = Z_target.shape

    device = esn.A.device
    dtype = torch.float64 if force_float64 else esn.A.dtype

    esn = esn.to(device=device, dtype=dtype).eval()
    Z_target = Z_target.to(device=device, dtype=dtype)

    if int(esn.d) != int(d):
        raise ValueError(f"esn.d={int(esn.d)} must match Z_target last dim d={int(d)}")

    # Precompute model reservoir states once
    _, Xhist = _sample_states_once(
        esn,
        T=T,
        N_model=N_model,
        dtype=dtype,
        device=device,
        xi=xi,
        eta=eta,
    )  # Xhist: (N_model,T,h)

    # Optimize W (copy so we don't mutate esn until the end)
    W = torch.nn.Parameter(esn.W.detach().clone().to(device=device, dtype=dtype))

    opt = torch.optim.LBFGS(
        [W],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
    )

    def mmd_loss_from_W() -> Tensor:
        Zm = Xhist @ W.T  # (N_model,T,d)

        if kernel_mode == "static":
            Xk = Z_target.reshape(N_target, -1)
            Yk = Zm.reshape(N_model, -1)
        else:
            Xk, Yk = Z_target, Zm

        # compute_mmd_loss is assumed to handle different batch sizes
        return compute_mmd_loss(kernel, Xk, Yk, lead_lag, lags)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = mmd_loss_from_W()
        loss.backward()
        return loss

    m0 = float(mmd_loss_from_W().detach().cpu())
    if verbose:
        print("MMD initial:", m0)

    opt.step(closure)

    m1 = float(mmd_loss_from_W().detach().cpu())
    if verbose:
        print("MMD final:", m1)

    # Write W back into esn
    with torch.no_grad():
        esn.W.copy_(W)

    return {
        "W_fit": W.detach().cpu(),
        "mmd_initial": m0,
        "mmd_final": m1,
        "T": int(T),
        "N_target": int(N_target),
        "N_model": int(N_model),
        "dtype": str(dtype),
    }


def compare_Ws(
    W_fit: Tensor,
    W_fixed: Tensor,
    *,
    title: str = "W_fit vs W_fixed",
    scatter: bool = True,
) -> Dict[str, float]:
    """
    Compares fitted W to a reference fixed W.
    Works for any (d,h) shape.
    """
    W_fit = W_fit.detach().cpu()
    W_fixed = W_fixed.detach().cpu()

    if W_fit.shape != W_fixed.shape:
        raise ValueError(f"Shape mismatch: W_fit {tuple(W_fit.shape)} vs W_fixed {tuple(W_fixed.shape)}")

    diff = W_fit - W_fixed

    mse = float((diff ** 2).mean())
    mse_ref = float((W_fixed ** 2).mean())
    rel_mse = float(mse / (mse_ref + 1e-12))

    fro = float(torch.linalg.norm(diff))
    fro_ref = float(torch.linalg.norm(W_fixed))
    rel_fro = float(fro / (fro_ref + 1e-12))

    a = W_fit.reshape(-1).numpy()
    b = W_fixed.reshape(-1).numpy()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")

    print(f"E[W_fixed^2]          = {mse_ref:.6g}")
    print(f"MSE(W_fit, W_fixed)   = {mse:.6g}")
    print(f"relative MSE          = {rel_mse:.6g}")
    print(f"||W_fixed||_F         = {fro_ref:.6g}")
    print(f"||W_fit||_F           = {float(torch.linalg.norm(W_fit)):.6g}")
    print(f"||W_fit-W_fixed||_F   = {fro:.6g}")
    print(f"relative Frobenius    = {rel_fro:.6g}")
    print(f"Corr(flattened)       = {corr:.6g}")

    if scatter:
        plt.figure(figsize=(5, 5))
        plt.scatter(b, a, s=10)
        plt.xlabel("W_fixed entries")
        plt.ylabel("W_fit entries")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return {
        "mse": mse,
        "mse_ref": mse_ref,
        "rel_mse": rel_mse,
        "fro": fro,
        "fro_ref": fro_ref,
        "rel_fro": rel_fro,
        "corr": corr,
    }
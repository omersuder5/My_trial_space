import torch
import sigkernel_ as ksig

def median_heuristic_sigma(
    X: torch.Tensor,
    Z: torch.Tensor,
    *,
    max_points: int = 2000,
    eps: float = 1e-12,
) -> float:
    """
    Median heuristic for RBF bandwidth on flattened paths.

    X, Z: (N, T, d) or already flattened (N, D)
    Returns sigma (not sigma^2).
    """
    # flatten to (N, D)
    if X.ndim == 3:
        Xf = X.reshape(X.shape[0], -1)
    elif X.ndim == 2:
        Xf = X
    else:
        raise ValueError("X must be (N,T,d) or (N,D)")

    if Z.ndim == 3:
        Zf = Z.reshape(Z.shape[0], -1)
    elif Z.ndim == 2:
        Zf = Z
    else:
        raise ValueError("Z must be (N,T,d) or (N,D)")

    U = torch.cat([Xf, Zf], dim=0)  # (2N, D)

    # optional subsample to keep pairwise distances manageable
    if U.shape[0] > max_points:
        idx = torch.randperm(U.shape[0], device=U.device)[:max_points]
        U = U[idx]

    # pairwise squared distances
    # D2_ij = ||u_i||^2 + ||u_j||^2 - 2 u_i^T u_j
    n2 = (U * U).sum(dim=1, keepdim=True)           # (M,1)
    D2 = n2 + n2.t() - 2.0 * (U @ U.t())            # (M,M)
    D2 = D2.clamp_min_(0.0)

    # take upper-triangular off-diagonal entries
    iu = torch.triu_indices(D2.shape[0], D2.shape[1], offset=1, device=U.device)
    d2_vals = D2[iu[0], iu[1]]

    med_d2 = torch.median(d2_vals)
    sigma2 = torch.clamp(med_d2, min=eps)
    sigma = torch.sqrt(sigma2).item()
    return sigma


# Voltera Kernel CV for lambda and tau
def gram_offdiag_cv(K: torch.Tensor, eps: float = 1e-12) -> float:
    """
    CV = std/mean over off-diagonal entries of a square Gram matrix.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")
    n = K.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=K.device)
    vals = K[mask]
    mean = vals.mean().abs().clamp_min(eps)
    std = vals.std(unbiased=False)
    return float((std / mean).item())

def volterra_cv_for_coeffs(
    X_ref: torch.Tensor, *,
    tau_coef: float,
    ld_coef: float,
    eps: float = 1e-12,
) -> float:
    kernel = ksig.kernels.VolterraKernel(tau_coef=tau_coef, ld_coef=ld_coef, eps=eps).fit(X_ref)
    # use Kxx on a subset for speed if needed: X_ref[:n_cal]
    K = kernel.compute_Gram(X_ref, X_ref)
    return gram_offdiag_cv(K, eps=eps)

def tune_tau_coef_for_cv(
    X_ref: torch.Tensor, *,
    ld_coef: float,
    cv_target: float = 0.05,
    tau_lo: float = 1e-2,
    tau_hi: float = 1.0,
    iters: int = 50,
    n_cal: int = None,
    eps: float = 1e-12,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    """
    Fix ld_coef, tune tau_coef to hit cv_target on K(X_ref,X_ref) off-diagonals.
    Returns (best_tau_coef, achieved_cv, kernel_spec).
    """
    X = X_ref[:n_cal] if n_cal is not None else X_ref
    if device is not None or dtype is not None:
        X = X.to(device=device or X.device, dtype=dtype or X.dtype)

    best = None
    lo, hi = float(tau_lo), float(tau_hi)

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        cv = volterra_cv_for_coeffs(X, tau_coef=mid, ld_coef=ld_coef, eps=eps)

        # store best by absolute error
        err = abs(cv - cv_target)
        if best is None or err < best[0]:
            # also keep the fitted spec for inspection
            ker = ksig.kernels.VolterraKernel(tau_coef=mid, ld_coef=ld_coef, eps=eps).fit(X)
            best = (err, mid, cv, ker.spec())

        # monotone heuristic: increase tau if CV too small
        if cv < cv_target:
            lo = mid
        else:
            hi = mid

    _, tau_best, cv_best, spec_best = best
    return tau_best, cv_best, spec_best

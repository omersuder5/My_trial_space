import torch

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

from __future__ import annotations

from typing import Any, Optional, Literal, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sigkernel_.loss import compute_mmd_loss

Tensor = torch.Tensor


def _sample_target(
    *,
    dataloader: Optional[DataLoader],
    dl_it: Optional[iter],
    target_generator: Optional[Any],
    batch_size: int,
    T: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if dataloader is not None:
        if dl_it is None:
            dl_it = iter(dataloader)
        try:
            batch = next(dl_it)
        except StopIteration:
            dl_it = iter(dataloader)
            batch = next(dl_it)

        X = batch[0] if isinstance(batch, (tuple, list)) else batch
        X = X.to(device=device, dtype=dtype)[:batch_size]

        if X.ndim != 3:
            raise ValueError("Dataloader must yield X with shape (B,T,d)")
        if X.shape[1] != T:
            raise ValueError(f"Data T mismatch: got {X.shape[1]}, expected {T}")
        if X.shape[2] != d:
            raise ValueError(f"Data d mismatch: got {X.shape[2]}, expected {d}")

        return X, dl_it

    if target_generator is not None:
        if hasattr(target_generator, "generate"):
            X = target_generator.generate(N=batch_size)
        else:
            X = target_generator(T=T, N=batch_size)

        X = X.to(device=device, dtype=dtype)

        if X.ndim != 3:
            raise ValueError("Target generator must return (B,T,d)")
        if X.shape != (batch_size, T, d):
            raise ValueError(f"Target generator returned {tuple(X.shape)}, expected ({batch_size},{T},{d})")

        return X, dl_it

    raise ValueError("Provide exactly one of dataloader or target_generator")


def train_ESN_MMD_mvp(
    esn,
    kernel: Any,
    *,
    kernel_mode: Literal["static", "sequential"],
    T: int,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
    lead_lag: bool = False,
    lags: int = 1,
    dataloader: Optional[DataLoader] = None,
    target_generator: Optional[Any] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, List[float]]:

    if (dataloader is None) == (target_generator is None):
        raise ValueError("Provide exactly one of dataloader or target_generator")

    if not isinstance(lags, int) or lags < 0:
        raise ValueError("lags must be a nonnegative int")

    if lead_lag and kernel_mode == "static":
        raise ValueError("lead_lag requires kernel_mode='sequential'")

    if device is None:
        device = esn.W.device

    # >>> FORCE ESN TO FLOAT64 <<<
    esn = esn.to(device=device, dtype=dtype)
    esn.train(True)

    opt = torch.optim.Adam([esn.W], lr=lr)
    dl_it = iter(dataloader) if dataloader is not None else None

    losses: List[float] = []
    for i in tqdm(range(epochs)):
        X, dl_it = _sample_target(
            dataloader=dataloader,
            dl_it=dl_it,
            target_generator=target_generator,
            batch_size=batch_size,
            T=T,
            d=esn.d,
            device=device,
            dtype=dtype,   # <<< USE dtype HERE
        )

        Z = esn(T=T, N=batch_size)  # now float64

        if kernel_mode == "static":
            Xk = X.reshape(batch_size, -1)
            Zk = Z.reshape(batch_size, -1)
        else:
            Xk, Zk = X, Z

        loss = compute_mmd_loss(kernel, Xk, Zk, lead_lag, lags)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.detach().cpu()))
        average_last_20_loss = np.mean(losses[-20:])
        if (i + 1) % 100 == 0:
            tqdm.write(f"Epoch {i+1}/{epochs}, Avg last 20: {average_last_20_loss:.6f}")

    return {"losses": losses, "average_last_20_loss": average_last_20_loss}


from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Literal, Dict, List, Deque, Tuple
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

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
) -> Tuple[Tensor, Optional[iter]]:
    if dataloader is not None:
        if dl_it is None:
            dl_it = iter(dataloader)
        try:
            batch = next(dl_it)
        except StopIteration:
            dl_it = iter(dataloader)
            batch = next(dl_it)

        X = batch[0] if isinstance(batch, (tuple, list)) else batch
        X = X.to(device=device, dtype=dtype)

        # strict shape checking for training stability
        if X.shape[0] != batch_size:
            raise ValueError("Dataloader batch_size mismatch. Use DataLoader(..., drop_last=True).")
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


def train_ESN_MMD(
    esn,
    kernel: Any,
    *,
    kernel_mode: Literal["static", "sequential"],
    T: int,
    epochs: int,
    batch_size: int,
    d: Optional[int] = None,                   # NEW: allow explicit d; defaults to esn.d
    dataloader: Optional[DataLoader] = None,
    target_generator: Optional[Any] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> Dict[str, Any]:
    """
    Supports either:
      - dataloader (real data), or
      - target_generator (oracle simulator)

    Exactly one must be provided.
    """
    # defaults
    lr: float = float(kwargs.get("lr", 1e-3))
    lead_lag: bool = bool(kwargs.get("lead_lag", False))
    lags: int = int(kwargs.get("lags", 1))

    out_dir: str = str(kwargs.get("out_dir", "./runs/esn_mmd"))
    run_name: str = str(kwargs.get("run_name", "default"))
    save_every: int = int(kwargs.get("save_every", 10))
    num_losses: int = int(kwargs.get("num_losses", 20))
    checkpoint: Optional[Dict[str, Any]] = kwargs.get("checkpoint", None)

    lr_factor: float = float(kwargs.get("lr_factor", 0.5))
    plateau_patience: int = int(kwargs.get("plateau_patience", 50))
    max_lr_drops: int = int(kwargs.get("max_lr_drops", 3))
    early_stopping_patience: int = int(kwargs.get("early_stopping_patience", 200))
    min_lr: float = float(kwargs.get("min_lr", 1e-6))

    # validate
    if (dataloader is None) == (target_generator is None):
        raise ValueError("Provide exactly one of dataloader or target_generator")
    if not isinstance(lags, int) or lags < 0:
        raise ValueError("lags must be a nonnegative int")
    if lead_lag and kernel_mode == "static":
        raise ValueError("lead_lag requires kernel_mode='sequential'")
    if not (0.0 < lr_factor < 1.0):
        raise ValueError("lr_factor must be in (0,1)")
    if plateau_patience <= 0 or early_stopping_patience <= 0:
        raise ValueError("patience values must be > 0")
    if max_lr_drops < 0:
        raise ValueError("max_lr_drops must be >= 0")
    if save_every <= 0 or num_losses <= 0:
        raise ValueError("save_every and num_losses must be > 0")

    if device is None:
        device = esn.W.device

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_tag = str(device).replace(":", "")  # e.g. "cuda0" or "cpu"

    run_id = f"{run_name}_{timestamp}_{device_tag}"
    run_path = Path(out_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    esn = esn.to(device=device, dtype=dtype)
    esn.train(True)

    # NEW: decide expected d (for dataloader or generator checks)
    d_expected = int(esn.d if d is None else d)

    opt = torch.optim.Adam([esn.W], lr=lr)
    dl_it = iter(dataloader) if dataloader is not None else None

    # state
    start_epoch = 0
    losses: List[float] = []
    avg_losses: List[float] = []
    last_k: Deque[float] = deque(maxlen=num_losses)
    best_avg_loss = float("inf")
    best_epoch = -1

    lr_drops_used = 0
    last_improve_epoch = -1

    # restore
    if checkpoint is not None:
        start_epoch = int(checkpoint["epoch"]) + 1
        losses = list(checkpoint.get("losses", []))
        last_k = deque(list(checkpoint.get("last_k_losses", [])), maxlen=num_losses)

        best_avg_loss = float(checkpoint.get("best_avg_loss", best_avg_loss))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        lr_drops_used = int(checkpoint.get("lr_drops_used", lr_drops_used))
        last_improve_epoch = int(checkpoint.get("last_improve_epoch", last_improve_epoch))

        if "esn_state_dict" in checkpoint:
            esn.load_state_dict(checkpoint["esn_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])

    pbar = tqdm(range(start_epoch, epochs), desc="train")
    for epoch in pbar:
        X, dl_it = _sample_target(
            dataloader=dataloader,
            dl_it=dl_it,
            target_generator=target_generator,
            batch_size=batch_size,
            T=T,
            d=d_expected,          # NEW: use d_expected
            device=device,
            dtype=dtype,
        )

        Z = esn(T=T, N=batch_size)  # (B,T,esn.d)

        # NEW: enforce ESN output dimension agrees with expected d
        if Z.ndim != 3 or Z.shape[1] != T or Z.shape[2] != d_expected:
            raise ValueError(f"ESN returned {tuple(Z.shape)}, expected ({batch_size},{T},{d_expected})")

        if kernel_mode == "static":
            Xk = X.reshape(batch_size, -1)
            Zk = Z.reshape(batch_size, -1)
        else:
            Xk, Zk = X, Z

        loss = compute_mmd_loss(kernel, Xk, Zk, lead_lag, lags)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        epoch_loss = float(loss.detach().cpu())
        losses.append(epoch_loss)

        last_k.append(epoch_loss)
        avg_k = float(np.mean(last_k)) if len(last_k) else epoch_loss
        avg_losses.append(avg_k)

        lr_now = opt.param_groups[0]["lr"]
        pbar.set_postfix(loss=epoch_loss, avg_k=avg_k, lr=lr_now, drops=lr_drops_used)

        # best model tracking (on smoothed avg_k)
        if avg_k < best_avg_loss:
            best_avg_loss = avg_k
            best_epoch = epoch
            last_improve_epoch = epoch
            torch.save(esn.state_dict(), run_path / "best_model.pt")

        # plateau => LR drop (limited)
        if last_improve_epoch >= 0:
            if (epoch - last_improve_epoch) >= plateau_patience and lr_drops_used < max_lr_drops:
                new_lr = max(lr_now * lr_factor, min_lr)
                if new_lr < lr_now:
                    for g in opt.param_groups:
                        g["lr"] = new_lr
                    lr_drops_used += 1
                    last_improve_epoch = epoch
                    tqdm.write(f"LR drop {lr_drops_used}/{max_lr_drops}: lr -> {new_lr:.3g}")

        early_stopping_bool = best_epoch >= 0 and (epoch - best_epoch) >= early_stopping_patience

        # checkpoint
        if ((epoch + 1) % save_every) == 0 or early_stopping_patience == 0 or early_stopping_bool:

            kernel_spec = {
                "kernel_mode": kernel_mode,
                "kernel_name": kernel.__class__.__name__,
                "kernel_params": {},
                "kernel_str": None,
            }

            # common case: RBF-like
            if hasattr(kernel, "sigma"):
                kernel_spec["kernel_params"]["sigma"] = float(kernel.sigma)

            # sigkernel SigKernel case
            if hasattr(kernel, "dyadic_order"):
                kernel_spec["kernel_params"]["dyadic_order"] = int(kernel.dyadic_order)
            if hasattr(kernel, "static_kernel"):
                sk = kernel.static_kernel
                kernel_spec["kernel_params"]["static_kernel_name"] = sk.__class__.__name__
                if hasattr(sk, "sigma"):
                    kernel_spec["kernel_params"]["static_sigma"] = float(sk.sigma)
            if hasattr(kernel, "n_levels"):
                    kernel_spec["kernel_params"]["n_levels"] = int(kernel.n_levels)

            kernel_spec["kernel_str"] = (
                f'{kernel_spec["kernel_name"]}(' +
                ", ".join(f"{k}={v}" for k, v in kernel_spec["kernel_params"].items()) +
                ")"
            )

            ckpt = {
                "epoch": epoch,
                "esn_state_dict": esn.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "losses": losses,
                "avg_losses": avg_losses,
                "best_avg_loss": best_avg_loss,
                "best_epoch": best_epoch,
                "last_k_losses": list(last_k),
                "lr_drops_used": lr_drops_used,
                "last_improve_epoch": last_improve_epoch,
                "config": {
                    "kernel_mode": kernel_mode,
                    "kernel_spec": kernel_spec,
                    "T": T,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "d": d_expected,
                    "dtype": str(dtype),
                    "device": str(device),
                    "generator_type": (
                        "dataloader" if dataloader is not None else "target_generator"
                    ),
                    # kwargs snapshot
                    "lr": lr,
                    "lead_lag": lead_lag,
                    "lags": lags,
                    "out_dir": out_dir,
                    "run_name": run_name,
                    "save_every": save_every,
                    "num_losses": num_losses,
                    "lr_factor": lr_factor,
                    "plateau_patience": plateau_patience,
                    "max_lr_drops": max_lr_drops,
                    "early_stopping_patience": early_stopping_patience,
                    "min_lr": min_lr,
                    "run_id": run_id,
                },
            }
            esn_spec = {
                "A": esn.A.detach().cpu(),
                "C": esn.C.detach().cpu(),
                "out_dim": int(esn.d),
                "activation": getattr(esn, "activation_name", "tanh"),
                "xi_scale": float(esn.xi_scale),
                "eta_scale": float(esn.eta_scale),
                "t_tilt": (None if esn.t_tilt is None else esn.t_tilt.detach().cpu()),
                "target_rho": float(kwargs.get("target_rho", 0.9)),  # optional, informative only
            }
            ckpt["esn_spec"] = esn_spec
            torch.save(ckpt, run_path / "checkpoint.pt")
            np.save(run_path / "losses.npy", np.asarray(losses, dtype=np.float64))
            np.save(run_path / "avg_losses.npy", np.asarray(avg_losses, dtype=np.float64))

        # early stop
        if best_epoch >= 0 and (epoch - best_epoch) >= early_stopping_patience:
            tqdm.write(
                f"Early stopping at epoch {epoch} "
                f"(best avg_{num_losses} {best_avg_loss:.6g} at epoch {best_epoch}, lr_drops_used={lr_drops_used})"
            )
            break

    torch.save(esn.state_dict(), run_path / "final_model.pt")
    np.save(run_path / "losses.npy", np.asarray(losses, dtype=np.float64))
    np.save(run_path / "avg_losses.npy", np.asarray(avg_losses, dtype=np.float64))

    return {
        "losses": losses,
        "avg_losses": avg_losses,
        "best_avg_loss": best_avg_loss,
        "best_epoch": best_epoch,
        "lr_drops_used": lr_drops_used,
        "final_lr": opt.param_groups[0]["lr"],
        "run_path": str(run_path),
        "run_path": str(run_path),
        "run_id": run_id,
    }

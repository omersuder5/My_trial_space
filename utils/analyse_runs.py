from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Literal
import re

import numpy as np
import pandas as pd
import torch

from generators.ESN import ESNGenerator


def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if (d is not None and key in d) else default


def _kernel_to_str(kernel: Any) -> str:
    # best-effort readable identifier
    cls = kernel.__class__.__name__
    # common sigma attributes
    sigma = getattr(kernel, "sigma", None)
    if sigma is not None:
        return f"{cls}(sigma={sigma})"
    # some kernels keep static kernel inside
    sk = getattr(kernel, "static_kernel", None)
    if sk is not None:
        return f"{cls}(static={sk.__class__.__name__})"
    return cls


def _parse_run_folder_name(name: str) -> Dict[str, Optional[str]]:
    # expected: <run_name>_YYYYMMDD_HHMMSS_<deviceTag>
    m = re.search(r"(?P<ts>\d{8}_\d{6})_(?P<dev>cpu|cuda\d+)$", name)
    if not m:
        return {"timestamp": None, "device_tag": None, "base_name": name}
    ts = m.group("ts")
    dev = m.group("dev")
    base = name[: m.start()]  # everything before timestamp
    if base.endswith("_"):
        base = base[:-1]
    return {"timestamp": ts, "device_tag": dev, "base_name": base}


def build_runs_df(
    runs_dir: str | Path = "./runs",
    *,
    recursive: bool = True,
    prefer: str = "checkpoint",  # "checkpoint" or "final"
    include_losses_tail: int = 0,  # 0 = don't store tail arrays
    map_location: str = "cpu",
) -> pd.DataFrame:
    """
    Scan runs_dir for saved runs and return a DataFrame with configs + key outcomes.

    Assumes each run folder may contain:
      - checkpoint.pt (preferred) with keys: epoch, losses, avg_losses, best_avg_loss, best_epoch, lr_drops_used, config
      - losses.npy / avg_losses.npy (optional, can be used if checkpoint missing)

    prefer:
      - "checkpoint": use checkpoint.pt if present
      - "final": fall back to losses.npy/avg_losses.npy more aggressively
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")

    run_folders = [p for p in runs_dir.glob("**/*") if p.is_dir()] if recursive else [p for p in runs_dir.iterdir() if p.is_dir()]

    rows: List[Dict[str, Any]] = []

    for run_path in run_folders:
        ckpt_path = run_path / "checkpoint.pt"
        losses_path = run_path / "losses.npy"
        avg_losses_path = run_path / "avg_losses.npy"

        if prefer == "checkpoint" and not ckpt_path.exists() and not losses_path.exists():
            continue
        if prefer == "final" and not losses_path.exists() and not ckpt_path.exists():
            continue

        row: Dict[str, Any] = {"run_path": str(run_path), "run_folder": run_path.name}
        row.update(_parse_run_folder_name(run_path.name))

        ckpt = None
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=map_location)
            except Exception as e:
                row["load_error"] = f"checkpoint load failed: {e}"
                rows.append(row)
                continue

        config = _safe_get(ckpt, "config", {}) if ckpt is not None else {}

        # config
        row["kernel_spec"] = _safe_get(config, "kernel_spec", None)
        row["T"] = _safe_get(config, "T", None)
        row["batch_size"] = _safe_get(config, "batch_size", None)
        row["d"] = _safe_get(config, "d", None)
        row["dtype"] = _safe_get(config, "dtype", None)
        row["device"] = _safe_get(config, "device", None)

        # training hyperparams
        row["lr"] = _safe_get(config, "lr", None)
        row["lr_factor"] = _safe_get(config, "lr_factor", None)
        row["plateau_patience"] = _safe_get(config, "plateau_patience", None)
        row["max_lr_drops"] = _safe_get(config, "max_lr_drops", None)
        row["early_stopping_patience"] = _safe_get(config, "early_stopping_patience", None)
        row["min_lr"] = _safe_get(config, "min_lr", None)
        row["num_losses"] = _safe_get(config, "num_losses", None)
        row["lead_lag"] = _safe_get(config, "lead_lag", None)
        row["lags"] = _safe_get(config, "lags", None)

        # outcomes
        row["epoch_last"] = _safe_get(ckpt, "epoch", None)
        row["best_avg_loss"] = _safe_get(ckpt, "best_avg_loss", None)
        row["best_epoch"] = _safe_get(ckpt, "best_epoch", None)
        row["lr_drops_used"] = _safe_get(ckpt, "lr_drops_used", None)

        # losses and derived summaries
        losses = _safe_get(ckpt, "losses", None)
        avg_losses = _safe_get(ckpt, "avg_losses", None)  # may not exist in older runs

        # fallback to npy files
        if losses is None and losses_path.exists():
            losses = np.load(losses_path).tolist()
        if avg_losses is None and avg_losses_path.exists():
            avg_losses = np.load(avg_losses_path).tolist()

        if losses is not None:
            row["n_epochs_ran"] = len(losses)
            row["final_loss"] = float(losses[-1])
        if avg_losses is not None:
            row["final_avg_loss"] = float(avg_losses[-1])

        if include_losses_tail > 0 and losses is not None:
            row["losses_tail"] = losses[-include_losses_tail:]
        if include_losses_tail > 0 and avg_losses is not None:
            row["avg_losses_tail"] = avg_losses[-include_losses_tail:]

        rows.append(row)

    df = pd.DataFrame(rows)

    # nice ordering if present
    preferred_cols = [
        "base_name", "timestamp", "device_tag", "run_folder",
        "kernel_spec", "T", "batch_size", "d",
        "lr", "lr_factor", "plateau_patience", "max_lr_drops", "early_stopping_patience", "num_losses",
        "best_avg_loss", "best_epoch", "final_avg_loss", "final_loss", "n_epochs_ran", "lr_drops_used",
        "run_path",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    losses = {
        "losses": losses,
        "avg_losses": avg_losses,
    }


    return df, losses


# Get generator ------------------------------------------------------------------
from pathlib import Path
from typing import Literal, Union
import torch

def load_esn_from_run(
    run_path: Union[str, Path],
    *,
    which: Literal["best", "final", "checkpoint"] = "best",
    map_location: Union[str, torch.device] = "cpu",
    dtype: torch.dtype | None = None,
):
    run_path = Path(run_path)

    # load checkpoint for spec
    ckpt = torch.load(run_path / "checkpoint.pt", map_location="cpu")
    spec = ckpt["esn_spec"]

    A = spec["A"]
    C = spec["C"]
    out_dim = int(spec["out_dim"])
    activation = spec.get("activation", "tanh")
    xi_scale = float(spec.get("xi_scale", 1.0))
    eta_scale = float(spec.get("eta_scale", 1.0))
    t_tilt = spec.get("t_tilt", None)

    esn = ESNGenerator(
        A=A,
        C=C,
        out_dim=out_dim,
        activation=activation,
        xi_scale=xi_scale,
        eta_scale=eta_scale,
        t_tilt=t_tilt,
        target_rho=0.9,  # ignored effectively since A already stored scaled
    )

    # load parameters
    if which == "best":
        state = torch.load(run_path / "best_model.pt", map_location="cpu")
        esn.load_state_dict(state)
    elif which == "final":
        state = torch.load(run_path / "final_model.pt", map_location="cpu")
        esn.load_state_dict(state)
    elif which == "checkpoint":
        esn.load_state_dict(ckpt["esn_state_dict"])
    else:
        raise ValueError("which must be 'best', 'final', or 'checkpoint'")

    # move to requested device/dtype
    esn = esn.to(map_location)
    if dtype is not None:
        esn = esn.to(dtype=dtype)
    esn.eval()
    return esn


def load_esn_from_df(df: pd.DataFrame, row: int, **load_kwargs):
    return load_esn_from_run(df.loc[row, "run_path"], **load_kwargs)
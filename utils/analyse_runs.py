from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Literal, Union
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from generators.ESN import ESNGenerator
from generators.synthetic_generators import ARMA, GARCH11, Noise
from statsmodels.tsa.stattools import acf as sm_acf


def load_runs_table(out_dir: str = "./runs") -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows = []
    loss_collection = []

    for run_path in sorted(out_dir.glob("*")):
        ckpt_path = run_path / "checkpoint.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")

        cfg = ckpt.get("config", {})
        kernel_spec = cfg.get("kernel_spec", None)

        esn_spec = ckpt.get("esn_spec", None)
        if esn_spec is None:
            esn_spec = cfg.get("esn_spec", None)

        target_spec = ckpt.get("target_spec", None)
        if target_spec is None:
            target_spec = cfg.get("target_spec", None)

        def _fmt_dict(d: dict | None, keys: list[str]) -> dict:
            if not isinstance(d, dict):
                return {k: None for k in keys}
            out = {}
            for k in keys:
                out[k] = d.get(k, None)
            return out

        base = {
            "run_id": run_path.name,
            "run_path": str(run_path),
            "epoch": int(ckpt.get("epoch", -1)),
            "best_epoch": int(ckpt.get("best_epoch", -1)),
            "best_avg_loss": float(ckpt.get("best_avg_loss", float("nan"))),
            "lr_drops_used": int(ckpt.get("lr_drops_used", 0)),
            "target_noise_spec": _fmt_dict(cfg.get("target_noise_spec", None), ["kind", "params"]),
        }

        # Kernel columns (compact)
        if isinstance(kernel_spec, dict):
            base["kernel_mode"] = kernel_spec.get("kernel_mode")
            base["kernel_spec"] = kernel_spec.get("kernel_str") or kernel_spec.get("kernel_name")
        else:
            base["kernel_mode"] = cfg.get("kernel_mode", None)
            base["kernel_spec"] = None

        # ESN columns (compact)
        if isinstance(esn_spec, dict):
            base.update({
                "esn_h": getattr(esn_spec.get("A", None), "shape", [None, None])[0] if hasattr(esn_spec.get("A", None), "shape") else None,
                "esn_m": getattr(esn_spec.get("C", None), "shape", [None, None])[1] if hasattr(esn_spec.get("C", None), "shape") else None,
                "esn_out_dim": esn_spec.get("out_dim", None),
                "esn_activation": esn_spec.get("activation", None),
                "esn_xi_scale": esn_spec.get("xi_scale", None),
                "esn_eta_scale": esn_spec.get("eta_scale", None),
                "esn_target_rho": esn_spec.get("target_rho", None),
                "esn_xi_ma_theta": esn_spec.get("xi_ma_theta", None).tolist() if isinstance(esn_spec.get("xi_ma_theta", None), torch.Tensor) else esn_spec.get("xi_ma_theta", None),
                "esn_quad_gain": esn_spec.get("quad_gain", None),
            })
        else:
            base.update({"esn_h": None, "esn_m": None, "esn_out_dim": None, "esn_activation": None, "esn_xi_scale": None, "esn_eta_scale": None, "esn_target_rho": None, "esn_xi_ma_theta": None, "esn_quad_gain": None})

        # Target generator columns (ARMA / GARCH friendly)
        if isinstance(target_spec, dict):
            base["target_name"] = target_spec.get("name", None)
            base["target_T"] = target_spec.get("T", None)
            base["target_p"] = target_spec.get("p", None)
            base["target_q"] = target_spec.get("q", None)
            base["target_phi"] = target_spec.get("phi", None)
            base["target_theta"] = target_spec.get("theta", None)
            base["target_omega"] = target_spec.get("omega", None)
            base["target_alpha"] = target_spec.get("alpha", None)
            base["target_beta"] = target_spec.get("beta", None)
            base["target_sigma2_0"] = target_spec.get("sigma2_0", None)
        else:
            base.update({"target_name": None, "target_T": None, "target_p": None, "target_q": None,
                         "target_phi": None, "target_theta": None, "target_omega": None, "target_alpha": None, "target_beta": None, "target_noise_spec": None, "target_sigma2_0": None})

        rows.append(base)

    df = pd.DataFrame(rows)

    # Nicely order columns
    col_order = [
        "run_id", "best_avg_loss", "best_epoch", "epoch", "lr_drops_used",
        "kernel_mode", "kernel_spec",
        "esn_h", "esn_m", "esn_out_dim", "esn_activation", "esn_xi_scale", "esn_eta_scale", "esn_target_rho", "esn_xi_ma_theta", "esn_quad_gain",
        "target_name", "target_T", "target_p", "target_q", "target_phi", "target_theta", "target_omega", "target_alpha", "target_beta", "target_noise_spec", "target_sigma2_0",
        "run_path",
    ]
    keep = [c for c in col_order if c in df.columns]
    df = df[keep].sort_values("best_avg_loss", ascending=True, na_position="last").reset_index(drop=True)

    return df

# load esn ------------------------------------------------------------------
def load_esn_from_run(
    run_path: Union[str, Path],
    *,
    which: Literal["best", "final", "checkpoint"] = "best",
    map_location: Union[str, torch.device] = "cpu",
    dtype: torch.dtype | None = None,
):
    run_path = Path(run_path)

    ckpt = torch.load(run_path / "checkpoint.pt", map_location="cpu")
    spec = ckpt["esn_spec"]

    A = spec["A"]
    C = spec["C"]
    out_dim = int(spec["out_dim"])
    activation = spec.get("activation", "tanh")
    xi_scale = float(spec.get("xi_scale", 1.0))
    eta_scale = float(spec.get("eta_scale", 1.0))
    t_tilt = spec.get("t_tilt", None)
    target_rho = float(spec.get("target_rho", 0.9))
    xi_ma_theta = spec.get("xi_ma_theta", None)
    quad_feedback = spec.get("quad_feedback", False)
    quad_gain = float(spec.get("quad_gain", 0.0)) if spec.get("quad_gain", None) is not None else 0.0

    esn = ESNGenerator(
        A=A,
        C=C,
        out_dim=out_dim,
        activation=activation,
        xi_scale=xi_scale,
        eta_scale=eta_scale,
        t_tilt=t_tilt,
        target_rho=target_rho,  # informative only; A already stored
        xi_ma_theta=xi_ma_theta,
        quad_feedback=quad_feedback,
        quad_gain=quad_gain,
    )

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

    esn = esn.to(device=map_location)
    if dtype is not None:
        esn = esn.to(dtype=dtype)
    esn.eval()
    return esn

def load_esn_from_df(df: pd.DataFrame, row: int, **load_kwargs):
    return load_esn_from_run(df.loc[row, "run_path"], **load_kwargs)


# ---------- Target generator reconstruction ----------
def build_noise_from_spec(spec: dict | None):
    if spec is None:
        return None
    return Noise(kind=spec["kind"], params=spec.get("params", None))

def build_target_from_spec(spec: dict):
    name = spec.get("name", None)
    if name is None:
        raise ValueError("Target spec has no 'name' field.")

    if name == "ARMA":
        T = int(spec["T"])
        p = int(spec["p"])
        q = int(spec["q"])
        phi = spec.get("phi", None)
        theta = spec.get("theta", None)
        d = int(spec.get("d", 1))
        return ARMA(T=T, p=p, q=q, phi=phi, theta=theta, d=d)

    if name == "GARCH11":
        T = int(spec["T"])
        omega = float(spec["omega"])
        alpha = float(spec["alpha"])
        beta = float(spec["beta"])
        sigma2_0 = float(spec.get("sigma2_0", 1.0))
        d = int(spec.get("d", 1))
        return GARCH11(T=T, omega=omega, alpha=alpha, beta=beta, d=d, sigma2_0=sigma2_0)

    raise ValueError(f"Unknown target generator name: {name}")

def load_target_and_noise_from_run(
    run_path,
    *,
    map_location="cpu",
    dtype=None,
):
    run_path = Path(run_path)
    ckpt = torch.load(run_path / "checkpoint.pt", map_location="cpu")

    cfg = ckpt["config"]

    if "target_spec" not in cfg:
        raise ValueError("Checkpoint does not contain 'target_spec'.")

    target = build_target_from_spec(cfg["target_spec"])

    noise = build_noise_from_spec(cfg.get("target_noise_spec", None))

    target = target.to(device=map_location)
    if dtype is not None:
        target = target.to(dtype=dtype)
    target.eval()

    return target, noise

def load_target_and_noise_from_df(df, row: int, **load_kwargs):
    return load_target_and_noise_from_run(df.loc[row, "run_path"], **load_kwargs)


# ACF analysis ----------------------------------------------------------------------
def acf_compare(
    *,
    esn,
    generator,
    N: int,
    T: int,
    lag: int = 40,
    noise=None,
    component: int = 0,
    show_example: bool = True,
):
    """
    Samples N paths of length T from:
      - target: generator.generate(...)
      - model:  esn(...)

    Plots ACF of x_t and of x_t^2, side by side (Target vs ESN).
    Returns the averaged ACF arrays.

    Assumes x_t is already the series of interest (no prices, no returns transform).
    """
    if T < 2:
        raise ValueError("Need T >= 2.")
    lag_eff = int(min(lag, T - 1))
    lags = np.arange(lag_eff + 1)

    esn.eval()
    generator.eval()

    with torch.no_grad():
        X = generator.generate(N=N, T=T, noise=noise).detach().cpu()
        Z = esn(T=T, N=N).detach().cpu()

    def mean_acf(paths_3d: torch.Tensor, square: bool) -> np.ndarray:
        A = np.empty((paths_3d.shape[0], lag_eff + 1), dtype=float)
        for i in range(paths_3d.shape[0]):
            x = paths_3d[i, :, component].numpy()
            if square:
                x = x * x
            A[i, :] = sm_acf(x, nlags=lag_eff, fft=True)
        return A.mean(axis=0)

    acf_X = mean_acf(X, square=False)
    acf_Z = mean_acf(Z, square=False)
    acf2_X = mean_acf(X, square=True)
    acf2_Z = mean_acf(Z, square=True)

    if show_example:
        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax[0].plot(X[0, :, component].numpy())
        ax[0].set_title("Target example path")
        ax[1].plot(Z[0, :, component].numpy())
        ax[1].set_title("ESN example path")
        ax[1].set_xlabel("t")
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    ax[0, 0].stem(lags, acf_X, basefmt=" ")
    ax[0, 0].set_title("Target ACF of x_t")
    ax[0, 1].stem(lags, acf_Z, basefmt=" ")
    ax[0, 1].set_title("ESN ACF of x_t")

    ax[1, 0].stem(lags, acf2_X, basefmt=" ")
    ax[1, 0].set_title("Target ACF of x_t^2")
    ax[1, 1].stem(lags, acf2_Z, basefmt=" ")
    ax[1, 1].set_title("ESN ACF of x_t^2")

    ax[1, 0].set_xlabel("lag")
    ax[1, 1].set_xlabel("lag")
    ax[0, 0].set_ylabel("acf")
    ax[1, 0].set_ylabel("acf")

    plt.tight_layout()
    plt.show()

    return {
        "lags": lags,
        "target_acf": acf_X,
        "esn_acf": acf_Z,
        "target_acf_sq": acf2_X,
        "esn_acf_sq": acf2_Z,
    }
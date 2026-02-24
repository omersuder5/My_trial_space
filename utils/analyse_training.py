from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from sigkernel_.loss import compute_mmd_loss


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from statsmodels.tsa.stattools import acf as sm_acf

def plot_loss_from_run(
    results: dict,
    kwargs: dict,
    *,
    plot_avg: bool = True,
    overlay_raw: bool = False,
    log_scale: bool = False,
    overlay_lr: bool = False,
    mark_best_and_drops: bool = False,
    title: str | None = None,
):
    run_path = Path(results["run_path"])

    losses = np.load(run_path / "losses.npy")
    avg_losses = np.load(run_path / "avg_losses.npy")

    if plot_avg:
        y_main = avg_losses
        ylabel = "avg loss"
    else:
        y_main = losses
        ylabel = "raw loss"

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(y_main, label=("avg loss" if plot_avg else "raw loss"))

    if overlay_raw:
        ax.plot(losses, alpha=0.4, label="raw loss")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Training loss: {run_path.name}"
    ax.set_title(title)

    # Optionally overlay LR and mark events from checkpoint
    if overlay_lr or mark_best_and_drops:
        ckpt_path = run_path / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")

            best_epoch = ckpt.get("best_epoch", None)
            best_loss = ckpt.get("best_avg_loss", None)

            if mark_best_and_drops and best_epoch is not None:
                ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label="best epoch")

    ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "losses": losses,
        "avg_losses": avg_losses,
        "best_epoch": best_epoch,
        "best_avg_loss": best_loss,
    }


def inspect_run_and_mmd(
    results: dict,
    kwargs: dict,
    *,
    n_examples: int = 5,
    kernel_mmd: object,
    T_plot: int | None = None,
    _T_mmd: int = None,
    _N_mmd: int = None,
    noise: object | None = None,   # optional, only used if your generator needs it
    use_best: bool = False,        # whether to load best model weights for inspection
):
    esn = kwargs["esn"]
    target_generator = kwargs.get("target_generator", None)
    dataloader = kwargs.get("dataloader", None)
    T_mmd = int(kwargs.get("T")) if _T_mmd is None else int(_T_mmd)
    N_mmd = int(kwargs.get("batch_size")) if _N_mmd is None else int(_N_mmd)

    if target_generator is None and dataloader is None:
        raise ValueError("kwargs must contain either target_generator or dataloader for this inspector.")

    if kernel_mmd is None:
        kernel = kwargs["kernel"]
        kernel_mode = kwargs["kernel_mode"]
    else:
        kernel = kernel_mmd
        if "sig" in kernel.__class__.__name__.lower() or "volt" in kernel.__class__.__name__.lower():
            kernel_mode = "sequential"
        else:
            kernel_mode = "static"
    lead_lag = bool(kwargs.get("lead_lag", False))
    lags = int(kwargs.get("lags", 1))
    dtype = kwargs.get("dtype", torch.float64)
    device = kwargs.get("device", esn.W.device)

    T0 = int(kwargs["T"] if T_plot is None else T_plot)

    # ---------------- load best model weights if requested ----------------
    if use_best:
        if results is None or "run_path" not in results:
            raise ValueError("use_best=True requires results['run_path']")

        run_path = Path(results["run_path"])
        best_path = run_path / "best_model.pt"
        ckpt_path = run_path / "checkpoint.pt"

        if best_path.exists():
            state = torch.load(best_path, map_location="cpu")
            esn.load_state_dict(state)
        elif ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "esn_state_dict" not in ckpt:
                raise ValueError("checkpoint.pt does not contain 'esn_state_dict'")
            esn.load_state_dict(ckpt["esn_state_dict"])
        else:
            raise FileNotFoundError(f"Neither {best_path} nor {ckpt_path} exists.")

    # move ESN after (potential) load
    esn = esn.to(device=device, dtype=dtype)
    esn.eval()

    # sample target and ESN for plotting
    with torch.no_grad():
        if hasattr(target_generator, "generate"):
            if noise is None:
                X_tgt = target_generator.generate(N=n_examples)
            else:
                X_tgt = target_generator.generate(N=n_examples, noise=noise)
        elif dataloader is not None:
            batch = next(iter(dataloader))
            X_tgt = batch[0][:n_examples]
        else:
            X_tgt = target_generator(T=T0, N=n_examples)

        X_tgt = X_tgt.to(dtype=dtype, device="cpu")

        Z_esn = esn(T=T0, N=n_examples).detach().cpu().to(dtype=dtype)

    fig, axes = plt.subplots(nrows=n_examples, ncols=2, figsize=(12, 1.2 * n_examples), sharex=True)
    if n_examples == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_examples):
        axes[i, 0].plot(X_tgt[i, :, 0].numpy())
        axes[i, 0].set_title(f"Target {i+1}")
        axes[i, 1].plot(Z_esn[i, :, 0].numpy())
        axes[i, 1].set_title(f"ESN {i+1}")
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    plt.tight_layout()
    plt.show()

    # fresh MMD estimate
    with torch.no_grad():
        if hasattr(target_generator, "generate"):
            if noise is None:
                X = target_generator.generate(N=N_mmd)
            else:
                X = target_generator.generate(N=N_mmd, noise=noise)
        elif dataloader is not None:
            batch = next(iter(dataloader))
            X = batch[0][:N_mmd]
        else:
            X = target_generator(T=T_mmd, N=N_mmd)

        X = X.to(device=device, dtype=dtype)
        Z = esn(T=T_mmd, N=N_mmd).to(device=device, dtype=dtype)

        if kernel_mode == "static":
            Xk = X.reshape(X.shape[0], -1)
            Zk = Z.reshape(Z.shape[0], -1)
        else:
            Xk, Zk = X, Z

        mmd = compute_mmd_loss(kernel, Xk, Zk, lead_lag, lags)

    mmd_value = float(mmd.detach().cpu())
    print("MMD:", mmd_value)
    if results is not None:
        return {"run_path": results["run_path"], "mmd": mmd_value}
    else:
        return {"mmd": mmd_value}

# ACF analysis function ----------------------------------------------------------------------
def acf_analysis(
    results: dict | None,
    kwargs: dict,
    *,
    T_acf: int | None = None,
    lag_acf: int = 40,
    N_paths: int = 100,
    component: int = 0,
    noise: object | None = None,
    use_best: bool = False,
    show_paths: bool = True,
):
    esn = kwargs["esn"]
    target_generator = kwargs.get("target_generator", None)
    dataloader = kwargs.get("dataloader", None)

    if target_generator is None and dataloader is None:
        raise ValueError("kwargs must contain either target_generator or dataloader.")

    dtype = kwargs.get("dtype", torch.float64)
    device = kwargs.get("device", esn.W.device)

    if T_acf is None:
        if "T" not in kwargs:
            raise ValueError("Provide T_acf or kwargs['T'].")
        T_acf = int(kwargs["T"])
    else:
        T_acf = int(T_acf)

    if T_acf < 2:
        raise ValueError("Need T_acf >= 2.")

    # load best model weights if requested
    if use_best:
        if results is None or "run_path" not in results:
            raise ValueError("use_best=True requires results['run_path']")
        run_path = Path(results["run_path"])
        best_path = run_path / "best_model.pt"
        ckpt_path = run_path / "checkpoint.pt"

        if best_path.exists():
            esn.load_state_dict(torch.load(best_path, map_location="cpu"))
        elif ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "esn_state_dict" not in ckpt:
                raise ValueError("checkpoint.pt does not contain 'esn_state_dict'")
            esn.load_state_dict(ckpt["esn_state_dict"])
        else:
            raise FileNotFoundError("No best_model.pt or checkpoint.pt found.")

    esn = esn.to(device=device, dtype=dtype)
    esn.eval()

    # sample target and esn paths: (N_paths, T_acf, d)
    with torch.no_grad():
        if dataloader is not None:
            batch = next(iter(dataloader))
            X_tgt = batch[0] if isinstance(batch, (tuple, list)) else batch
            N_paths = min(X_tgt.shape[0], N_paths)
            T_acf = min(X_tgt.shape[1], T_acf)
            if X_tgt.ndim != 3:
                raise ValueError(f"dataloader must yield (B,T,d); got {tuple(X_tgt.shape)}")
            X_tgt = X_tgt[:N_paths, :T_acf, :].detach().cpu().to(dtype=dtype)
                
        else:
            if hasattr(target_generator, "generate"):
                # assumes your generator supports overriding T
                X_tgt = target_generator.generate(N=N_paths, T=T_acf, noise=noise)
            else:
                X_tgt = target_generator(T=T_acf, N=N_paths, noise=noise)
            if not isinstance(X_tgt, torch.Tensor) or X_tgt.ndim != 3:
                raise ValueError("target_generator must return a Tensor with shape (N,T,d).")
            X_tgt = X_tgt.detach().cpu().to(dtype=dtype)

        Z_esn = esn(T=T_acf, N=N_paths).detach().cpu().to(dtype=dtype)

    # Effective max lag cannot exceed T-1
    lag_eff = min(int(lag_acf), int(T_acf - 1))
    lags = np.arange(lag_eff + 1)

    def mean_acf_over_paths(X_3d: torch.Tensor, *, square: bool) -> np.ndarray:
        """
        Input X_3d: (N_paths, T, d)
        For each path i, take the 1D series x_i(t) = X_3d[i, :, component].

        If square=False:
          compute ACF of x_i(t).

        If square=True:
          compute ACF of x_i(t)^2.

        Then average the resulting ACF vectors over i=1..N_paths.

        statsmodels.tsa.stattools.acf returns:
          a[0] = 1 by definition
          a[k] ~ sample Corr(x_t, x_{t-k})
        It automatically subtracts the sample mean before computing autocovariances.
        """
        A = np.empty((X_3d.shape[0], lag_eff + 1), dtype=float)
        for i in range(X_3d.shape[0]):
            x = X_3d[i, :, component].numpy()
            if square:
                x = x * x
            # nlags cannot exceed len(x)-1, but lag_eff already respects T-1
            A[i, :] = sm_acf(x, nlags=lag_eff, fft=True)
        return A.mean(axis=0)

    # ACF(x) and ACF(x^2) for target and esn
    acf_tgt = mean_acf_over_paths(X_tgt, square=False)
    acf_esn = mean_acf_over_paths(Z_esn, square=False)
    acf2_tgt = mean_acf_over_paths(X_tgt, square=True)
    acf2_esn = mean_acf_over_paths(Z_esn, square=True)

    if show_paths:
        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax[0].plot(X_tgt[0, :, component].numpy())
        ax[0].set_title("Target example path")
        ax[1].plot(Z_esn[0, :, component].numpy())
        ax[1].set_title("ESN example path")
        ax[1].set_xlabel("t")
        plt.tight_layout()
        plt.show()

    # ACF plots: 2 rows (x and x^2), 2 cols (target and ESN)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    ax[0, 0].stem(lags, acf_tgt, basefmt=" ")
    ax[0, 0].set_title("Target ACF of x_t")
    ax[0, 1].stem(lags, acf_esn, basefmt=" ")
    ax[0, 1].set_title("ESN ACF of x_t")

    ax[1, 0].stem(lags, acf2_tgt, basefmt=" ")
    ax[1, 0].set_title("Target ACF of x_t^2")
    ax[1, 1].stem(lags, acf2_esn, basefmt=" ")
    ax[1, 1].set_title("ESN ACF of x_t^2")

    for j in range(2):
        ax[1, j].set_xlabel("lag")
    for i in range(2):
        ax[i, 0].set_ylabel("acf")

    plt.tight_layout()
    plt.show()

    return {
        "run_path": None if results is None else results.get("run_path"),
        "T_acf": int(T_acf),
        "lag_eff": int(lag_eff),
        "N_paths": int(N_paths),
        "component": int(component),
        "acf": {
            "lags": lags,
            "target_x": acf_tgt,
            "esn_x": acf_esn,
            "target_x2": acf2_tgt,
            "esn_x2": acf2_esn,
        },
    }
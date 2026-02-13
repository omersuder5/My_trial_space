from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from sigkernel_.loss import compute_mmd_loss


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


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
            lr_drops_used = ckpt.get("lr_drops_used", None)
            # We did not store full LR history, only current state,
            # so LR overlay is limited unless you later log LR per epoch.

            if mark_best_and_drops and best_epoch is not None:
                ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label="best epoch")

            # If you later store lr history, we can plot it here.
            if overlay_lr:
                print("Note: LR history not stored per epoch yet. Overlay skipped.")

    ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "losses": losses,
        "avg_losses": avg_losses,
    }


def inspect_run_and_mmd(
    results: dict,
    kwargs: dict,
    *,
    n_examples: int = 5,
    T_plot: int | None = None,
    T_mmd: int = 200,
    N_mmd: int = 50,
    noise: object | None = None,   # optional, only used if your generator needs it
):
    esn = kwargs["esn"]
    target_generator = kwargs.get("target_generator", None)
    if target_generator is None:
        raise ValueError("kwargs must contain target_generator for this inspector.")

    kernel = kwargs["kernel"]
    kernel_mode = kwargs["kernel_mode"]
    lead_lag = bool(kwargs.get("lead_lag", False))
    lags = int(kwargs.get("lags", 1))
    dtype = kwargs.get("dtype", torch.float64)
    device = kwargs.get("device", esn.W.device)

    T0 = int(kwargs["T"] if T_plot is None else T_plot)

    esn = esn.to(device=device, dtype=dtype)
    esn.eval()

    # sample target and ESN for plotting
    with torch.no_grad():
        if hasattr(target_generator, "generate"):
            if noise is None:
                X_tgt = target_generator.generate(N=n_examples)
            else:
                X_tgt = target_generator.generate(N=n_examples, noise=noise)
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
        else:
            X = target_generator(T=T_mmd, N=N_mmd)

        X = X.to(device=device, dtype=dtype)
        Z = esn(T=T_mmd, N=N_mmd).to(device=device, dtype=dtype)

        if kernel_mode == "static":
            Xk = X.reshape(N_mmd, -1)
            Zk = Z.reshape(N_mmd, -1)
        else:
            Xk, Zk = X, Z

        mmd = compute_mmd_loss(kernel, Xk, Zk, lead_lag, lags)

    mmd_value = float(mmd.detach().cpu())
    print("MMD:", mmd_value)

    return {"run_path": results["run_path"], "mmd": mmd_value}

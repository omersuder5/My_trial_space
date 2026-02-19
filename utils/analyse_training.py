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

    # ---------------- NEW: load best model weights if requested ----------------
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

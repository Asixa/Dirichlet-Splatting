"""Shared experiment I/O, synthetic acquisition and plots."""

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .evaluation import write_ply


def parser(experiment, presets):
    p = argparse.ArgumentParser(description=f"{experiment}: an experiment on the dsplat library")
    p.set_defaults(experiment=experiment)
    p.add_argument("--preset", choices=list(presets), default="demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int)
    p.add_argument("--ridge", type=float)
    p.add_argument("--output")
    return p


def output(args, **config):
    """Create the output folder and record the arguments and configuration."""
    path = Path(args.output or f"outputs/{args.experiment}_{args.preset}_seed{args.seed}")
    path.mkdir(parents=True, exist_ok=True)
    _json(path / "configuration.json", dict(arguments=vars(args), **config))
    return path


def save_result(path, state, initial, metrics):
    values = {name: value.cpu().numpy() for name, value in state.params.items()}
    values.update({f"initial_{name}": value.cpu().numpy() for name, value in initial.items()})
    values.update(
        coefficients=state.coefficients.cpu().numpy(), prediction=state.prediction.cpu().numpy()
    )
    np.savez_compressed(path / "reconstruction.npz", **values)
    if values["centers"].shape[1] == 3:
        write_ply(path / "reconstruction.ply", values["centers"], values.get("normals"))
    _json(path / "history.json", state.history)
    _json(path / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def report(entry):
    if entry["iteration"] % 5 == 0:
        print(
            f"DSFW {entry['iteration']:4d} NMSE {entry['nmse']:.6g} "
            f"replaced={entry.get('replaced', 0)} LM={entry.get('lm_accepted', 0)}",
            flush=True,
        )


def describe(scan):
    """Print the dimensions of a scan and of the chirp behind it."""
    config = scan.fmcw_config
    apertures, bins = scan.data.shape
    print(f"scan       {str(scan.data.dtype).removeprefix('torch.')} [{apertures}, {bins}]: "
          f"{apertures} apertures x {bins} positive range bins")  # fmt: skip
    print(f"chirp      {config.fc_hz / 1e9:g} GHz carrier, {config.bandwidth_hz / 1e9:g} GHz "
          f"bandwidth, {config.adc_samples} ADC samples, FFT length {config.n_fft}, "
          f"{config.range_bin_spacing_m * 1000:.2f} mm per range bin", flush=True)  # fmt: skip


def select_diverse(scores, candidates, count, min_separation):
    """Greedy best-first indices at least min_separation apart (NumPy arrays)."""
    selected, kept = [], []
    for index in np.argsort(scores, kind="stable")[::-1]:
        point = candidates[index]
        if kept and np.linalg.norm(np.stack(kept) - point, axis=1).min() < min_separation:
            continue
        selected.append(int(index))
        kept.append(point)
        if len(selected) >= count:
            break
    return selected


def _json(path, value):
    text = json.dumps(value, indent=2, default=lambda v: asdict(v) if is_dataclass(v) else str(v))
    path.write_text(text, encoding="utf-8")


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_signals(path, target, prediction, history):
    plt = _figure()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ref = max(float(np.max(np.abs(target))), 1e-30)
    for ax, value, title in zip(
        axes[:2], (target, prediction), ("Target", "Reconstruction"), strict=True
    ):
        ax.imshow(
            20 * np.log10(np.maximum(np.abs(value) / ref, 1e-3)),
            vmin=-60,
            vmax=0,
            cmap="magma",
            origin="lower",
        )
        ax.set_title(title + " (dB)")
    axes[2].semilogy(np.maximum(history, 1e-16))
    axes[2].set(title="Complex measurement NMSE", xlabel="Iteration")
    fig.tight_layout()
    fig.savefig(path / "result.png", dpi=160)
    plt.close(fig)


def plot_spectra(path, target, prediction, history, extent):
    """Square-root magnitude of two upsampled spectra, so the sidelobes around each peak show."""
    plt = _figure()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    top = float(np.abs(target).max())
    for ax, value, title in zip(
        axes[:2], (target, prediction), ("Target", "Reconstruction"), strict=True
    ):
        ax.imshow(np.sqrt(np.abs(value).T / top), vmin=0, vmax=1, cmap="turbo", origin="lower",
                  extent=(0, extent, 0, extent), interpolation="bilinear")  # fmt: skip
        ax.set(title=title + " |Y|^(1/2)", xlabel="bin", ylabel="bin")
    axes[2].semilogy(np.maximum(history, 1e-16))
    axes[2].set(title="Complex measurement NMSE", xlabel="Iteration")
    fig.tight_layout()
    fig.savefig(path / "result.png", dpi=160)
    plt.close(fig)


def plot_plane(path, centers, amplitudes, reference):
    plt = _figure()
    count = 2 if reference is not None else 1
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4))
    axes = np.atleast_1d(axes)
    color = amplitudes / max(float(amplitudes.max()), 1e-30)
    axes[0].scatter(
        centers[:, 0] * 1000,
        centers[:, 1] * 1000,
        c=color,
        s=12 + 50 * color,
        vmin=0,
        vmax=1,
        cmap="magma",
        edgecolors="none",
    )
    axes[0].set_title("Dirichlet surfels + DSFW")
    if reference is not None:
        axes[1].imshow(reference.T, origin="lower", extent=(-40, 40, -40, 40), cmap="magma")
        axes[1].set_title("Reference support")
    for ax in axes:
        ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    fig.tight_layout()
    fig.savefig(path / "geometry.png", dpi=180)
    plt.close(fig)


def plot_bunny(path, ours, reference, initial, iterations):
    plt = _figure()
    panels = [
        (reference, "Reference"),
        (initial, "Initialization"),
        (ours, f"After {iterations} iterations"),
    ]
    fig = plt.figure(figsize=(4.6 * len(panels), 8))
    center = np.array([0, 0, 0.35])
    limit = max(35.0, *(float(np.abs(points - center).max()) * 1050 for points, _ in panels))
    for i, (points, title) in enumerate(panels, 1):
        p = (points[:: max(1, len(points) // 8000)] - center) * 1000
        style = dict(
            s=max(2.5, min(35, 800 / len(p))),
            c=p[:, 1],
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
            linewidths=0,
        )
        ax = fig.add_subplot(2, len(panels), i, projection="3d")
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], depthshade=False, **style)
        ax.set_title(f"{title} | {len(points):,} points", fontsize=10, pad=12)
        ax.view_init(elev=18, azim=-62)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 1))
        ax.set(
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            zlim=(-limit, limit),
            xlabel="x (mm)",
            ylabel="z - 350 (mm)",
            zlabel="y (mm)",
        )
        ax = fig.add_subplot(2, len(panels), len(panels) + i)
        ax.scatter(p[:, 0], p[:, 1], **style)
        ax.set(
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            xlabel="x (mm)",
            ylabel="y (mm)",
            title="Front projection",
        )
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path / "result.png", dpi=180, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

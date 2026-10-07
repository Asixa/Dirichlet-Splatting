"""2D spectrum fitting: tones along a shape, synthesized independently, recovered by DSFW."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dsplat import DSFWConfig, ParameterConfig, SpectralGrid, SpectralModel, fit
from dsplat.kernels import spectral as kernel

from .common import output, parser, plot_spectra, report, save_result
from .scenes import outline

PRESETS = {
    "smoke": (24, 4, 14),
    "demo": (64, 20, 80),
    "paper": (64, 80, 200),
}  # side, tones, steps
UPSAMPLE = 4  # figure samples per bin


def measure(scene, side, count, window, seed):
    """Spectrum of count complex tones synthesized in double precision, then FFT'd.

    Returns the spectrum [side, side], the tone positions [count, 2] in bins and the
    windowed samples, whose zero-padded FFT draws the spectrum between bins.
    """
    torch.manual_seed(seed)
    if scene == "random":
        truth = 2 + torch.rand((count, 2), device="cuda") * (side - 4)
    else:
        xy = torch.tensor(outline(scene, count), dtype=torch.float32, device="cuda")
        truth = side / 2 + 0.4 * side * xy
    amplitudes = torch.polar(
        torch.ones(count, device="cuda"), 2 * torch.pi * torch.rand(count, device="cuda")
    )
    t = torch.arange(side, device="cuda", dtype=torch.float64)
    tone = torch.zeros((side, side), device="cuda", dtype=torch.complex128)
    for c, b in zip(truth.double(), amplitudes.to(torch.complex128), strict=True):
        tone += b * torch.exp(2j * torch.pi * (c[0] * t[:, None] + c[1] * t[None, :]) / side)
    if window == "rect":
        taper = torch.ones(side, device="cuda", dtype=torch.float64)
    else:
        taper = getattr(torch, window + "_window")(
            side, periodic=True, device="cuda", dtype=torch.float64
        )
    samples = tone * taper[:, None] * taper[None] / taper.sum().square()
    target = torch.fft.fftn(samples).to(torch.complex64)
    print(f"scene      {count} unit tones along a {scene}, {window} window")
    print(f"samples    complex128 [{side}, {side}] -> spectrum complex64 [{side}, {side}]")
    return target, truth, samples


def run(args):
    side, count, steps = PRESETS[args.preset]
    grid = SpectralGrid((side, side), window=args.window)
    target, truth, samples = measure(args.scene, side, count, args.window, args.seed)
    params = {"centers": torch.rand_like(truth) * side}
    print(f"atoms      {count} fitted: centers [{count}, 2] in bins, complex amplitudes [{count}]")
    parameters = {"centers": ParameterConfig(period=grid.shape)}
    config = DSFWConfig(
        iterations=args.iterations or steps,
        ridge=1e-8 if args.ridge is None else args.ridge,
        coefficients="complex",
        replacement="utility",
        min_separation=0.35,
        lm_solver="dense",
        lm_steps=3,
        lm_rows=grid.size,
        lm_gain_ratio=0,
        screen_apertures=0,
        trust_radius=0.75,
        seed=args.seed,
    )
    path = output(args, grid=grid, strategy=config)
    initial = {k: v.clone() for k, v in params.items()}
    state = fit(
        SpectralModel(grid), target, params, config,
        candidates={"centers": grid.bins()}, parameters=parameters, callback=report,
    )  # fmt: skip
    difference = state.params["centers"].double()[:, None] - truth.double()[None]
    difference -= side * torch.round(difference / side)
    distance = difference.norm(dim=-1).cpu().numpy()
    i, j = linear_sum_assignment(distance)
    metrics = dict(
        nmse=state.history[-1]["nmse"],
        center_rmse_bins=float(np.sqrt(np.mean(distance[i, j] ** 2))),
    )
    save_result(path, state, initial, metrics)
    # The figure shows the spectra between bins: the zero-padded FFT of the
    # synthesized samples, and the fitted atoms evaluated at the same fractional bins.
    fine = UPSAMPLE * side
    axis = torch.arange(fine, device="cuda", dtype=torch.float32) / UPSAMPLE
    bins = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    fitted = kernel.render(grid, state.params["centers"], state.coefficients, bins)
    plot_spectra(
        path,
        torch.fft.fftn(samples, s=(fine, fine)).cpu().numpy(),
        fitted.reshape(fine, fine).cpu().numpy(),
        [r["nmse"] for r in state.history],
        side,
    )
    return path


if __name__ == "__main__":
    p = parser("signal2d", PRESETS)
    p.add_argument("--window", choices=["rect", "hann", "hamming"], default="rect")
    p.add_argument("--scene", choices=["star", "square", "letter_A", "random"], default="star")
    print(run(p.parse_args()).resolve())

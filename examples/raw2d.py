"""A planar FMCW acquisition, expressed entirely by parameter constraints."""

from math import isqrt

import torch

from dsplat import DSFWConfig, FMCWConfig, FMCWModel, ParameterConfig, fit
from dsplat.fmcw import migration
from dsplat.groundtruth import planar_array, synthesize

from .common import (
    describe,
    output,
    parser,
    plot_plane,
    plot_signals,
    report,
    save_result,
    select_diverse,
)
from .evaluation import geometry_metrics
from .scenes import plane_scene

# side, candidate grid resolution, surfels, steps, migration padding
PRESETS = {"smoke": (24, 16, 24, 12, 8), "demo": (48, 36, 96, 40, 4), "paper": (96, 56, 192, 80, 2)}


def measure(scene, side, resolution):
    """FMCW scan of a planar target from a side x side raster at a quarter wavelength.

    Returns the scan, the reference centers [N, 3] and the target mask [resolution]^2.
    """
    sensor = FMCWConfig(bandwidth_hz=4e9, adc_samples=128, pad_factor=2, sweep_time_s=20e-6)
    truth, normals, weights, mask = plane_scene(scene, resolution, device="cuda")
    weights = torch.full_like(weights, 1 / len(weights) ** 0.5)
    scan = synthesize(
        planar_array(side, sensor.lambda_m / 4, device="cuda"), truth, normals, weights, sensor
    )
    print(f"scene      {scene}: {len(truth)} reference surfels at z = 0.35 m")
    print(f"aperture   {side} x {side} raster at {sensor.lambda_m / 4 * 1000:.3f} mm pitch")
    describe(scan)
    return scan, truth, mask


def run(args):
    side, resolution, budget, steps, padding = PRESETS[args.preset]
    scan, truth, mask = measure(args.scene, side, resolution)
    axis = torch.linspace(-0.04, 0.04, resolution, device="cuda")
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([x.flatten(), y.flatten(), torch.full_like(x.flatten(), 0.35)], -1)
    if args.initialization == "migration":
        # Every preset covers the same 80 mm ROI; smaller apertures need more zero
        # padding to keep that FFT field of view.
        scores = (
            migration(scan, points, (isqrt(len(scan.data)),) * 2, padding=padding)
            .abs()
            .cpu()
            .numpy()
        )
        centers = points[select_diverse(scores, points.cpu().numpy(), budget, 0.002)].clone()
    else:
        generator = torch.Generator().manual_seed(args.seed)
        centers = (torch.rand((budget, 3), generator=generator) * 0.08 - 0.04).cuda()
        centers[:, 2] = 0.35
    normals = torch.zeros_like(centers)
    normals[:, 2] = -1
    area = (0.08 / (resolution - 1)) ** 2
    params = dict(centers=centers, normals=normals, areas=centers.new_full((len(centers),), area))
    parameters = dict(
        centers=ParameterConfig(
            trainable=(True, True, False),
            scale=0.001,
            lower=(-0.04, -0.04, 0.35),
            upper=(0.04, 0.04, 0.35),
        ),
        normals=ParameterConfig(trainable=False, normalize=True),
        areas=ParameterConfig(trainable=False),
    )
    config = DSFWConfig(
        iterations=args.iterations or steps,
        ridge=(1e-14 if args.ridge is None else args.ridge) * area,
        coefficients="complex",
        replacement="utility",
        min_separation=0.0005,
        lm_solver="dense",
        lm_steps=2,
        lm_rows=128 * scan.data.shape[1],
        lm_gain_ratio=0,
        screen_apertures=0,
        seed=args.seed,
    )
    print(f"surfels    {budget} fitted: centers [{budget}, 3] with z fixed, "
          f"normals and areas fixed, complex reflectivities [{budget}]")  # fmt: skip
    path = output(args, sensor=scan.fmcw_config, strategy=config, parameters=parameters)
    initial = {k: v.clone() for k, v in params.items()}
    # Omitted candidate fields are frozen and copy the replaced slot.
    state = fit(
        FMCWModel(scan), scan.data, params, config,
        candidates=dict(centers=points), parameters=parameters, callback=report,
    )  # fmt: skip
    recovered = state.params["centers"].cpu().numpy()
    metrics = dict(nmse=state.history[-1]["nmse"], objective=state.objective)
    metrics.update(geometry_metrics(recovered, truth.cpu().numpy()))
    save_result(path, state, initial, metrics)
    plot_plane(path, recovered, state.coefficients.abs().cpu().numpy(), mask.cpu().numpy())
    plot_signals(
        path,
        scan.data.cpu().numpy(),
        state.prediction.cpu().numpy(),
        [r["nmse"] for r in state.history],
    )
    return path


if __name__ == "__main__":
    p = parser("raw2d", PRESETS)
    p.add_argument(
        "--scene",
        choices=["square", "letter_A", "letter_T", "letter_H", "grid", "two_slabs"],
        default="letter_A",
    )
    p.add_argument("--initialization", choices=["migration", "random"], default="migration")
    print(run(p.parse_args()).resolve())

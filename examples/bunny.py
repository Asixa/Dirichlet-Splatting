"""3D bunny from three orthogonal views: random start, spatial search and merged replacement.

measure synthesizes the scan from the mesh; run reconstructs the surfels from it. The
defaults are the fast configuration used for the 6000-surfel runs.
"""

import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree

from dsplat import (
    CertificateSearchConfig,
    DSFWConfig,
    FMCWConfig,
    FMCWModel,
    ParameterConfig,
    SpatialSearchConfig,
    fit,
)
from dsplat.groundtruth import planar_array, synthesize

from .common import describe, output, parser, plot_bunny, report, save_result
from .evaluation import geometry_metrics
from .scenes import sample_surface, view_axes
from .visibility import aperture_visibility_sources

# reference points, pooled candidates per step, steps, refined peaks, fully refined peaks
PRESETS = {
    "smoke": (64, 16, 10, 64, 8),
    "demo": (512, 64, 60, 64, 8),
    "paper": (6000, 1024, 58, 1024, 64),
}
LOWER, UPPER = (-0.048, -0.048, 0.302), (0.048, 0.048, 0.398)  # metres
VIEWS, SIDE, PITCH = 3, 128, 0.0025  # orthogonal arrays of SIDE x SIDE apertures, metres
MESH = Path(__file__).resolve().parent.parent / "assets" / "bunny_60mm.ply"  # metres, y up


def measure(points, seed):
    """FMCW scan of reference surfels sampled on the bunny surface the arrays see.

    Returns the scan and the reference centers, normals and amplitudes [points, ...].
    """
    positions = torch.cat(
        [planar_array(SIDE, PITCH, axis, device="cuda") for axis in view_axes(VIEWS)]
    )
    sources = aperture_visibility_sources(positions.cpu().numpy(), VIEWS)
    mesh = trimesh.load(MESH, force="mesh", process=False)
    truth, normals, weights = sample_surface(mesh, points, seed, sources=sources)
    scan = synthesize(
        positions,
        *(torch.tensor(v, device="cuda") for v in (truth, normals, weights)),
        FMCWConfig(),
    )
    print(f"scene      {points} reference surfels on the visible bunny surface")
    print(f"arrays     {VIEWS} views x {SIDE} x {SIDE} apertures at {PITCH * 1000:g} mm pitch")
    describe(scan)
    return scan, truth, normals, weights


def run(args):
    points, batch, steps, peaks, full_peaks = PRESETS[args.preset]
    surfels = args.surfels or points
    scan, truth, truth_normals, weights = measure(points, args.seed)
    parameters = dict(
        centers=ParameterConfig(scale=0.0005, lower=LOWER, upper=UPPER),
        normals=ParameterConfig(scale=0.1, normalize=True),
        areas=ParameterConfig(trainable=False),
    )
    generator = torch.Generator().manual_seed(args.seed)
    lower, upper = torch.tensor(LOWER), torch.tensor(UPPER)
    initial = dict(
        centers=(lower + (upper - lower) * torch.rand(surfels, 3, generator=generator)).cuda(),
        normals=F.normalize(torch.randn(surfels, 3, generator=generator), dim=1).cuda(),
        areas=torch.full((surfels,), 4 * math.pi * 0.03**2 / surfels, device="cuda"),
    )
    print(f"surfels    {surfels} fitted: centers [{surfels}, 3], normals [{surfels}, 3], "
          f"areas [{surfels}] fixed, real reflectivities [{surfels}]")  # fmt: skip
    config = DSFWConfig(
        iterations=args.iterations or steps, ridge=args.ridge or 0.0, batch=batch, seed=args.seed
    )
    search = SpatialSearchConfig(
        full_peaks=full_peaks, continuous=CertificateSearchConfig(peaks=peaks)
    )
    path = output(
        args, sensor=scan.fmcw_config, strategy=config, search=search, parameters=parameters
    )
    start = perf_counter()

    def progress(entry):
        report(entry)
        if entry["iteration"] % 5 == 0:
            print(f"  elapsed {perf_counter() - start:.1f} s", flush=True)

    state = fit(
        FMCWModel(scan),
        scan.data,
        initial,
        config,
        search,
        parameters=parameters,
        callback=progress,
    )
    seconds = perf_counter() - start
    recovered = state.params["centers"].cpu().numpy()
    nearest = cKDTree(recovered).query(truth)[0]
    metrics = dict(
        nmse=state.history[-1]["nmse"],
        seconds=seconds,
        coverage_1mm=float((nearest < 0.001).mean()),
        **geometry_metrics(recovered, truth),
    )
    save_result(path, state, initial, metrics)
    np.savez_compressed(
        path / "reference.npz", points=truth, normals=truth_normals, weights=weights
    )
    plot_bunny(path, recovered, truth, initial["centers"].cpu().numpy(), state.iteration)
    return path


if __name__ == "__main__":
    p = parser("bunny", PRESETS)
    p.add_argument("--surfels", type=int, help="fitted surfels; default: the reference point count")
    print(run(p.parse_args()).resolve())

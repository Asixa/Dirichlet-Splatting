"""Bindings of surfel.slang and correlate.slang: oriented surfels seen as FMCW range profiles.

config is an FMCWConfig, positions the sensor positions [A, 3]. weights are the
per-surfel complex amplitudes, reflectivity times sqrt(area); the area chain is in
dsplat.surfel. surfel.slang renders in cotangent form (profiles, JVP, VJP, dictionary
rows); correlate.slang evaluates correlations, normal-bank scores and the explicit
Jacobian. Both share the scattering of scattering.slang.
"""

import torch
from torch.autograd.function import once_differentiable

from . import blocks, interleaved, load

PAIR_BYTES = 12  # amplitude (2 floats) and peak bin of one aperture-surfel pair
GEOMETRY_BYTES = 1 << 30  # workspace bound for the pair geometry of one launch
FILL_THREADS = 250_000  # below this, the surfel loop is split into segments
MAX_SEGMENTS = 32


def sensor_args(config):
    """Chirp arguments of the kernels in surfel.slang."""
    return dict(
        num_bins=config.num_range_bins,
        n_fft=config.n_fft,
        samples=config.adc_samples,
        k0_per_meter=1 / config.range_bin_spacing_m,
        fc=config.fc_hz,
        slope=config.slope_hz_s,
        t_start=config.adc_start_time_s,
        lambda_m=config.lambda_m,
    )


def _direct_args(config):
    """Chirp arguments of the kernels in correlate.slang."""
    args = sensor_args(config)
    del args["samples"]
    return dict(args, half_order=(config.adc_samples - 1) / 2)


def _launch(config, apertures, count):
    """Module, block and grid for one thread per aperture, bin tile and surfel segment.

    The surfel loop is split into segments only when the launch would otherwise have
    too few threads to fill the GPU.
    """
    pad = config.pad_factor
    tile = pad * -(-16 // pad)
    tiles = -(-config.num_range_bins // tile)
    width = min(tiles, 256)
    rows = max(1, 256 // width)
    segments = min(MAX_SEGMENTS, count, max(1, -(-FILL_THREADS // (apertures * tiles))))
    grid = (-(-tiles // width), -(-apertures // rows), segments)
    return load("surfel", PAD=pad, TILE=tile), (width, rows, 1), grid


def _profiles(config, positions, centers, normals, weights, fresnel_n):
    """Range profiles [A, K, 2] = sum_s weights_s * column_s.

    pair_geometry stores amplitude and peak bin per aperture-surfel pair; render_bins
    gives one thread an aperture, a tile of bins and a segment of surfels. Segment sums
    are added here in a fixed order.
    """
    bins, count = config.num_range_bins, len(centers)
    # Bounded workspace; int32 pair indices and the grid-y limit of one launch.
    chunk = min(GEOMETRY_BYTES // (PAIR_BYTES * count), (2**31 - 1) // count, 65535, len(positions))
    chunk = -(-len(positions) // -(-len(positions) // chunk))  # equal chunks fill the GPU
    output = centers.new_empty((len(positions), bins, 2))
    amplitude = centers.new_empty((chunk, count, 2))
    peak = centers.new_empty((chunk, count))
    geometry = {k: v for k, v in sensor_args(config).items() if k not in ("num_bins", "samples")}
    for start in range(0, len(positions), chunk):
        part = positions[start : start + chunk].contiguous()
        module, block, grid = _launch(config, len(part), count)
        module.pair_geometry(
            scan_positions=part,
            centers=centers,
            normals=normals,
            amplitude=amplitude,
            peak=peak,
            num_apertures=len(part),
            num_surfels=count,
            fresnel_n=fresnel_n,
            **geometry,
        ).launchRaw(**blocks(len(part) * count))
        target = output[start : start + len(part)]
        segments = grid[2]
        partial = (
            target.unsqueeze(1)
            if segments == 1
            else output.new_empty((len(part), segments, bins, 2))
        )
        module.render_bins(
            amplitude=amplitude,
            peak=peak,
            weights=weights,
            output=partial,
            num_apertures=len(part),
            num_surfels=count,
            num_bins=bins,
            n_fft=config.n_fft,
            samples=config.adc_samples,
            segment=-(-count // segments),
        ).launchRaw(blockSize=block, gridSize=grid)
        if segments > 1:
            target.copy_(partial.sum(1))
    return output


def _surfels(positions, centers, normals, weights, fresnel_n):
    return dict(
        scan_positions=positions.contiguous(),
        centers=centers.contiguous(),
        normals=normals.contiguous(),
        weights=interleaved(weights),
        num_apertures=len(positions),
        num_surfels=len(centers),
        fresnel_n=fresnel_n,
    )


def vjp(config, positions, centers, normals, weights, upstream, fresnel_n):
    """Gradients of Re(conj(upstream) . profiles) for centers, normals and complex weights."""
    count = len(centers)
    # One partial result per surfel and 32-aperture warp; summed here in double.
    warps = -(-len(positions) // 32)
    partial = centers.new_empty((count, warps, 8))
    module, _, _ = _launch(config, len(positions), count)
    module.vjp_bins(
        **_surfels(positions, centers, normals, weights, fresnel_n),
        upstream=interleaved(upstream),
        output=partial,
        **sensor_args(config),
    ).launchRaw(blockSize=(32, 4, 1), gridSize=(warps, -(-count // 4), 1))
    total = partial.double().sum(1).float()
    return total[:, :3], total[:, 3:6], torch.view_as_complex(total[:, 6:].contiguous())


def jvp(
    config,
    positions,
    centers,
    normals,
    weights,
    delta_centers,
    delta_normals,
    delta_weights,
    fresnel_n,
):
    """Directional derivative of the profiles [A, K]."""
    count = len(centers)
    module, block, grid = _launch(config, len(positions), count)
    partial = centers.new_empty((len(positions), grid[2], config.num_range_bins, 2))
    module.jvp_bins(
        **_surfels(positions, centers, normals, weights, fresnel_n),
        delta_centers=delta_centers.contiguous(),
        delta_normals=delta_normals.contiguous(),
        delta_weights=interleaved(delta_weights),
        output=partial,
        segment=-(-count // grid[2]),
        **sensor_args(config),
    ).launchRaw(blockSize=block, gridSize=grid)
    return torch.view_as_complex(partial[:, 0] if grid[2] == 1 else partial.sum(1))


class _Render(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, normals, weights, positions, config, fresnel_n):
        ctx.save_for_backward(centers, normals, weights, positions)
        ctx.config, ctx.fresnel_n = config, fresnel_n
        return _profiles(config, positions, centers, normals, weights, fresnel_n)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        centers, normals, weights, positions = ctx.saved_tensors
        upstream = torch.view_as_complex(grad.contiguous())
        complex_weights = torch.view_as_complex(weights)
        gc, gn, gw = vjp(
            ctx.config, positions, centers, normals, complex_weights, upstream, ctx.fresnel_n
        )
        return gc, gn, torch.view_as_real(gw), None, None, None


def render(config, positions, centers, normals, weights, fresnel_n):
    """Range profiles [A, K]; differentiable in centers, normals and weights."""
    output = _Render.apply(
        centers.contiguous(),
        normals.contiguous(),
        interleaved(weights),
        positions,
        config,
        fresnel_n,
    )
    return torch.view_as_complex(output)


def design(config, positions, centers, normals, areas, fresnel_n, real):
    """Dictionary rows of whole apertures, including sqrt(area).

    Complex rows are [A * K, S]. real gives [2 * A * K, S] ordered bin by bin, row
    (2 k + part) * A + a, so one product projects every aperture onto a row basis.
    """
    count, rows = len(centers), len(positions) * config.num_range_bins
    output = centers.new_empty((2 * rows, count) if real else (rows, 2 * count))
    module, _, _ = _launch(config, len(positions), count)
    module.design_bins(
        scan_positions=positions.contiguous(),
        centers=centers.contiguous(),
        normals=normals.contiguous(),
        areas=areas.contiguous(),
        output=output,
        num_apertures=len(positions),
        num_surfels=count,
        fresnel_n=fresnel_n,
        complex_entries=not real,
        **sensor_args(config),
    ).launchRaw(blockSize=(128, 1, 1), gridSize=(-(-count // 128), len(positions), 1))
    return output if real else torch.view_as_complex(output.view(rows, count, 2))


def jacobian(config, positions, centers, normals, fresnel_n):
    """Columns [A, K, S] and their center and normal derivatives [A, K, S, 3], without area."""
    out = centers.new_empty((len(positions), config.num_range_bins, len(centers), 2))
    jac = centers.new_empty((*out.shape[:-1], 6, 2))
    load("correlate").surfel_design_jacobian(
        scan_positions=positions.contiguous(),
        centers=centers.contiguous(),
        normals=normals.contiguous(),
        output=out,
        jacobian=jac,
        num_apertures=len(positions),
        num_surfels=len(centers),
        fresnel_n=fresnel_n,
        **_direct_args(config),
    ).launchRaw(**blocks(out.numel() // 2))
    jac = torch.view_as_complex(jac)
    return torch.view_as_complex(out), jac[..., :3], jac[..., 3:]


def correlate(config, positions, centers, normals, residual, fresnel_n):
    """conj(column) . residual and |column|^2 per candidate, without area."""
    count = len(centers)
    output = centers.new_zeros((3, count))
    load("correlate").surfel_correlate(
        scan_positions=positions.contiguous(),
        candidates=centers.contiguous(),
        cand_normals=normals.contiguous(),
        residual_re=residual.real.contiguous(),
        residual_im=residual.imag.contiguous(),
        corr_re=output[0],
        corr_im=output[1],
        norm2=output[2],
        num_apertures=len(positions),
        num_candidates=count,
        fresnel_n=fresnel_n,
        **_direct_args(config),
    ).launchRaw(blockSize=(64, 4, 1), gridSize=(-(-count // 64), -(-len(positions) // 4), 1))
    return torch.complex(output[0], output[1]), output[2]


def correlate_orientations(
    config, positions, centers, normals, bank, residual, fresnel_n, table=None
):
    """Correlations and norms [S, 1 + bank] for the current normal and every bank normal.

    table, an fmcw.RangeTable, replaces the per-bin sum by FFT-interpolated range
    correlations; it only ranks proposals. Without area.
    """
    output = centers.new_zeros((len(centers), len(bank) + 1, 3))
    load("correlate").surfel_correlate_orientations(
        scan_positions=positions.contiguous(),
        centers=centers.contiguous(),
        normals=normals.contiguous(),
        bank=bank.contiguous(),
        residual_re=residual.real.contiguous(),
        residual_im=residual.imag.contiguous(),
        output=output,
        num_apertures=len(positions),
        num_surfels=len(centers),
        num_normals=len(bank),
        use_table=table is not None,
        range_table=centers.new_empty((1, 1, 2)) if table is None else table.correlation,
        norm_table=centers.new_empty(1) if table is None else table.norm,
        oversample=1 if table is None else table.oversample,
        fresnel_n=fresnel_n,
        **_direct_args(config),
    ).launchRaw(blockSize=(32, 4, 1), gridSize=(-(-len(positions) // 32), -(-len(centers) // 4), 1))
    return torch.complex(output[..., 0], output[..., 1]), output[..., 2]


def correlate_gradient(config, positions, centers, normals, residual, fresnel_n, include_normals):
    """Correlation, norm and their center (and normal) derivatives [S, 3 or 6], without area."""
    width = 6 if include_normals else 3
    output = centers.new_zeros((len(centers), 3 + 3 * width))
    load("correlate").surfel_correlate_gradient(
        scan_positions=positions.contiguous(),
        centers=centers.contiguous(),
        normals=normals.contiguous(),
        residual_re=residual.real.contiguous(),
        residual_im=residual.imag.contiguous(),
        output=output,
        num_apertures=len(positions),
        num_surfels=len(centers),
        fresnel_n=fresnel_n,
        include_normals=include_normals,
        **_direct_args(config),
    ).launchRaw(blockSize=(32, 4, 1), gridSize=(-(-len(positions) // 32), -(-len(centers) // 4), 1))
    return (
        torch.complex(output[:, 0], output[:, 1]),
        output[:, 2],
        torch.complex(output[:, 3 : 3 + width], output[:, 3 + width : 3 + 2 * width]),
        output[:, 3 + 2 * width :],
    )

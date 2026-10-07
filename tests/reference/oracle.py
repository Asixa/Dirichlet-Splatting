"""Independent float64 PyTorch oracles for the Slang kernels."""

import torch

from dsplat.groundtruth import _echoes

from .kernel import dirichlet, windowed_kernel


def spectral_design(grid, centers, bins):
    """Dictionary [M, S] and its center derivative [M, S, D] in float64."""
    centers, bins = centers.double(), bins.double()
    parts = []
    for d, (n, f) in enumerate(zip(grid.samples, grid.shape, strict=True)):
        delta = bins[:, None, d] - centers[None, :, d]
        k, dk = windowed_kernel(delta, n, f, grid.window, grid.beta)
        if grid.radius:
            inside = (delta - f * torch.round(delta / f)).abs() <= grid.radius
            k, dk = k * inside, dk * inside
        parts.append((k, dk))
    a = torch.ones((len(bins), len(centers)), dtype=torch.complex128, device=centers.device)
    for k, _ in parts:
        a = a * k
    derivatives = []
    for d, (_, dk) in enumerate(parts):
        value = -dk
        for other, (k, _) in enumerate(parts):
            if other != d:
                value = value * k
        derivatives.append(value)
    return a, torch.stack(derivatives, -1)


def fmcw_columns(positions, centers, normals, config, fresnel_n=0.0):
    """Range-profile columns [A, K, S] in float64, without area."""
    distance, amplitude = _echoes(
        positions.double(), centers.double(), normals.double(), config, fresnel_n
    )
    bins = torch.arange(config.num_range_bins, device=centers.device, dtype=torch.float64)
    delta = bins[None, :, None] - distance[:, None, :] / config.range_bin_spacing_m
    return dirichlet(delta, config.adc_samples, config.n_fft)[0] * amplitude[:, None, :]

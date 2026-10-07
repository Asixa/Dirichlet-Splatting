"""Synthetic ground truth: planar arrays and FMCW scans of known surfels.

synthesize follows the FMCW signal chain itself: every surfel echo is a beat tone
sampled by the ADC, summed in double precision and range-FFT'd. It never evaluates
the Dirichlet kernel, so a fit to its scans does not reuse the forward model.
"""

import math

import numpy as np
import torch

from .fmcw import Scan


def planar_array(side, pitch, axis=(0.0, 0.0, 1.0), standoff=0.35, *, device="cpu"):
    """side x side aperture positions [side^2, 3] in metres, pitch apart.

    The array faces along axis toward the point (0, 0, standoff); the default is the
    plane z = 0 facing +z.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    source = np.array([0.0, 0.0, 1.0])
    cross = np.cross(source, axis)
    dot = source @ axis
    if np.linalg.norm(cross) < 1e-10:
        rotation = np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        x, y, z = cross
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + skew + skew @ skew / (1 + dot)
    t = (np.arange(side) - (side - 1) / 2) * pitch
    x, y = np.meshgrid(t, t, indexing="ij")
    base = np.stack([x.ravel(), y.ravel(), np.zeros(side * side)], -1)
    center = np.array([0, 0, standoff])
    return torch.tensor(center + (base - center) @ rotation.T, dtype=torch.float32, device=device)


def _echoes(aperture_positions, surfel_centers, surfel_normals, config, fresnel_n=0.0):
    """Range [A, S] and complex echo amplitude [A, S] of each aperture-surfel pair."""
    diff = aperture_positions[:, None, :] - surfel_centers[None, :, :]
    distance = diff.norm(dim=-1).clamp(min=1e-9)
    cosine = (diff / distance[..., None] * surfel_normals[None]).sum(-1).clamp(min=0)
    amplitude = cosine.square() * config.lambda_m / (8 * math.pi * distance)
    if fresnel_n > 1:
        root = torch.sqrt(fresnel_n * fresnel_n - 1 + cosine.square())
        amplitude = amplitude * (root - cosine) / (root + cosine)
    tau = 2 * distance / config.c0_m_s
    phase = (
        2
        * math.pi
        * (config.fc_hz * tau + config.slope_hz_s * tau * (config.adc_start_time_s - 0.5 * tau))
    )
    return distance, amplitude * torch.exp(1j * phase)


@torch.no_grad()
def synthesize(
    aperture_positions, surfel_centers, surfel_normals, surfel_amplitudes, config, fresnel_n=0.0
):
    """Scan [apertures, range bins] of surfels with amplitudes reflectivity * sqrt(area).

    Positions and normals in metres [., 3]; config is an FMCWConfig. Blocks of 256
    apertures and 512 surfels bound the double-precision workspace.
    """
    time = (
        torch.arange(config.adc_samples, device=surfel_centers.device, dtype=torch.float64)
        / config.sample_rate_hz
    )
    centers, normals = surfel_centers.double(), surfel_normals.double()
    amplitudes = surfel_amplitudes.to(torch.complex128)
    profiles = []
    for start in range(0, len(aperture_positions), 256):
        block = aperture_positions[start : start + 256].double()
        adc = torch.zeros(
            (len(block), config.adc_samples), device=centers.device, dtype=torch.complex128
        )
        for s in range(0, len(centers), 512):
            distance, echo = _echoes(
                block, centers[s : s + 512], normals[s : s + 512], config, fresnel_n
            )
            beat = 2 * config.slope_hz_s * distance / config.c0_m_s
            adc += (
                (echo * amplitudes[s : s + 512])[:, :, None]
                * torch.exp(2j * math.pi * beat[:, :, None] * time)
            ).sum(1)
        spectrum = torch.fft.fft(adc, n=config.n_fft)[:, : config.num_range_bins]
        profiles.append((spectrum / config.adc_samples).to(torch.complex64))
    return Scan(torch.cat(profiles), aperture_positions, config)

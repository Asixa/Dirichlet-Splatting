"""Double-precision composed range/Doppler/azimuth renderer, differentiable by autograd."""

import torch

from dsplat.spectral import SpectralGrid
from dsplat.surfel import Sensor, fresnel_te, project

from .oracle import spectral_design


def render_scene(
    positions, normals, areas, reflectivity, velocities, impedance, sensor: Sensor,
    *, origin=None, gamma=1.0, window="rect",
):  # fmt: skip
    origin = positions.new_zeros(3) if origin is None else origin
    delta = origin - positions
    r = delta.norm(dim=-1).clamp(min=1e-9)
    n = torch.nn.functional.normalize(normals, dim=-1)
    cos = (n * delta / r[:, None]).sum(-1).clamp(min=0)
    coeff = reflectivity * areas.sqrt() * cos.square() * fresnel_te(cos, impedance) * r.pow(-gamma)
    coeff = coeff * torch.exp(4j * torch.pi * r / sensor.wavelength)
    grid = SpectralGrid(sensor.shape, window=window)
    bins = grid.bins(positions.device).double()
    a, _ = spectral_design(grid, project(positions, velocities, sensor, origin), bins)
    return (a @ coeff).reshape(grid.shape)

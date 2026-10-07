"""Small CUDA problems shared by the tests; every production path is Slang on CUDA."""

import pytest
import torch

from dsplat import FMCWConfig, FMCWModel, ParameterConfig, Scan, SpectralGrid, SpectralModel

# Without CUDA only the example-helper tests can run.
if not torch.cuda.is_available():
    collect_ignore = [
        "test_spectral.py",
        "test_surfel.py",
        "test_fmcw.py",
        "test_solver.py",
        "test_search.py",
        "test_examples.py",
    ]


def close(actual, expected, tolerance=2e-5):
    """Agreement within tolerance times the largest expected magnitude."""
    expected = expected.to(actual.device, actual.dtype)
    atol = tolerance * float(expected.abs().max())
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


def aperture_grid(side=8, pitch=0.004):
    t = (torch.arange(side, dtype=torch.float32) - (side - 1) / 2) * pitch
    x, y = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([x.flatten(), y.flatten(), torch.zeros(side * side)], -1).cuda()


def surfels(count, seed=0, spread=0.01):
    """Surfels near (0, 0, 0.3) facing the apertures at z = 0."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([0, 0, 0.3]) + spread * (
        2 * torch.rand(count, 3, generator=generator) - 1
    )
    normals = torch.nn.functional.normalize(
        torch.tensor([0.0, 0, -1]) + 0.3 * torch.randn(count, 3, generator=generator), dim=1
    )
    areas = torch.full((count,), 1e-4)
    return {k: v.cuda() for k, v in dict(centers=centers, normals=normals, areas=areas).items()}


def fmcw_specs(trainable_normals=True):
    return dict(
        centers=ParameterConfig(scale=0.0005, lower=(-0.02, -0.02, 0.28), upper=(0.02, 0.02, 0.32)),
        normals=ParameterConfig(trainable=trainable_normals, scale=0.1, normalize=True),
        areas=ParameterConfig(trainable=False),
    )


@pytest.fixture
def fmcw():
    """An 8x8-aperture scan of five surfels with real reflectivities."""
    config = FMCWConfig()
    positions = aperture_grid()
    params = surfels(5)
    coefficients = torch.linspace(0.5, 1.5, 5, device="cuda")
    empty = Scan(
        torch.zeros((len(positions), config.num_range_bins), dtype=torch.complex64, device="cuda"),
        positions,
        config,
    )
    data = FMCWModel(empty).render(params, coefficients)
    return FMCWModel(Scan(data, positions, config)), params, coefficients


@pytest.fixture
def spectral():
    """Three complex tones on a 24 x 20 periodic grid."""
    grid = SpectralGrid((24, 20))
    centers = torch.tensor([[3.3, 4.1], [12.0, 15.0], [18.4, 7.9]], device="cuda")  # one on a bin
    coefficients = torch.tensor([1 + 0.5j, -0.7 + 0.2j, 0.4 - 0.9j], device="cuda")
    model = SpectralModel(grid)
    return model, {"centers": centers}, coefficients

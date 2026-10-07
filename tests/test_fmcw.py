"""FMCW measurement model: row basis, range gate and FFT range table."""

import pytest
import torch
from conftest import aperture_grid, close, surfels

from dsplat import FMCWConfig, FMCWModel, Scan
from dsplat.fmcw import response_basis


@pytest.mark.parametrize("pad", [2, 4])
def test_row_basis_preserves_the_normal_equations(pad):
    config = FMCWConfig(pad_factor=pad)
    data = torch.zeros((36, config.num_range_bins), dtype=torch.complex64, device="cuda")
    model = FMCWModel(Scan(data, aperture_grid(6), config))
    a = model.design(surfels(6), slice(0, model.size))
    spacing = config.range_bin_spacing_m
    reduced = response_basis(config, 0.28 / spacing, 0.33 / spacing, "cuda").reduce(a)
    close(reduced.H @ reduced, a.H @ a, 1e-4)


def test_range_gate_keeps_points_at_measured_ranges(fmcw):
    model, params, _ = fmcw
    away = params["centers"] + torch.tensor([0, 0, 0.05], device="cuda")
    points = torch.cat((params["centers"], away))
    keep = model.range_gate(points, model.scan.data, 0.1, margin_cells=1.0, min_fraction=0.9)
    assert keep[:5].all() and not keep[5:].any()


def test_fft_range_table_approximates_the_exact_proposal_scores(fmcw):
    model, params, _ = fmcw
    residual = model.scan.data
    bank = torch.nn.functional.normalize(torch.randn(6, 3, device="cuda"), dim=1)
    exact, norm = model.correlate_orientations(params, residual, bank)
    table = model.range_table(residual, 8)
    approximate, approximate_norm = model.correlate_orientations(params, residual, bank, table)
    close(approximate, exact, 3e-2)
    close(approximate_norm, norm, 3e-2)

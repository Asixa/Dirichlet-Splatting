"""Candidate grids, peak selection, range gating and certificate refinement."""

import math

import torch
from conftest import aperture_grid, fmcw_specs, surfels

from dsplat import DSFWConfig, FMCWConfig, FMCWModel, Scan, SpatialSearchConfig, initialize
from dsplat.certificate import (
    CertificateSearchConfig,
    candidate_grid,
    certificate,
    search_certificate,
    select_peaks,
)
from dsplat.spatial import propose


def brute_force_peaks(scores, centers, count, separation):
    scores, chosen = scores.clone(), []
    for _ in range(count):
        index = int(scores.argmax())
        if not torch.isfinite(scores[index]):
            break
        chosen.append(index)
        scores[(centers - centers[index]).norm(dim=1) < separation] = -torch.inf
    return chosen


def test_grid_corners_and_peaks():
    grid = candidate_grid([0.0, -1.0], [1.0, 1.0], 0.3)
    assert grid.shape == (5 * 8, 2) and grid.min(0).values.tolist() == [0, -1]
    generator = torch.Generator().manual_seed(1)
    centers = torch.rand(5000, 3, generator=generator).cuda()
    scores = torch.rand(5000, generator=generator).cuda()
    for count in (1, 10, 200):
        assert select_peaks(scores, centers, count, 0.1).tolist() == brute_force_peaks(
            scores, centers, count, 0.1
        )


def test_certificate_gradient_matches_finite_differences(fmcw):
    model, params, _ = fmcw
    residual = model.scan.data
    probe = {**params, "centers": params["centers"] + 2e-4}
    _, gradient = certificate(model, probe, residual, "real", gradient=True, normals=True)
    for axis in range(3):
        shift = torch.zeros_like(probe["centers"])
        shift[:, axis] = 1e-5  # a step float32 resolves at 0.3 m
        plus = certificate(model, {**probe, "centers": probe["centers"] + shift}, residual, "real")
        minus = certificate(model, {**probe, "centers": probe["centers"] - shift}, residual, "real")
        numeric = (plus - minus) / 2e-5
        torch.testing.assert_close(
            gradient[:, axis], numeric, rtol=2e-2, atol=1e-2 * float(gradient.abs().max())
        )


def test_continuous_refinement_reaches_a_single_reflector(fmcw):
    model, params, coefficients = fmcw
    one = {k: v[:1] for k, v in params.items()}
    residual = model.render(one, coefficients[:1])
    start = {k: v.repeat(8, *([1] * (v.ndim - 1))) for k, v in one.items()}
    start["centers"] = start["centers"] + 4e-4 * torch.randn_like(start["centers"])
    specs = fmcw_specs()
    params_out, scores = search_certificate(
        model, residual, start, specs["centers"], CertificateSearchConfig(peaks=8, separation=0), "real",
        normal_spec=specs["normals"],
    )  # fmt: skip
    assert float((params_out["centers"][0] - one["centers"][0]).norm()) < 5e-5
    assert scores[0] > 0.99


def test_spatial_search_proposes_the_true_reflector():
    config = FMCWConfig()
    positions = aperture_grid(8, 0.02)  # a 14 cm aperture resolves millimetres at 0.3 m
    truth = surfels(1, seed=5)
    empty = Scan(
        torch.zeros((64, config.num_range_bins), dtype=torch.complex64, device="cuda"),
        positions,
        config,
    )
    data = FMCWModel(empty).render(truth, torch.ones(1, device="cuda"))
    model = FMCWModel(Scan(data, positions, config))
    # A back-facing start renders nothing, so the residual is exactly the reflector.
    start = {**truth, "normals": torch.tensor([[0.0, 0.0, 1.0]], device="cuda")}
    dsfw_config = DSFWConfig(batch=1)
    state = initialize(model, data, start, dsfw_config, parameters=fmcw_specs())
    search = SpatialSearchConfig(
        proposal_apertures=64,
        refine_apertures=0,
        cache_peaks=0,
        continuous=CertificateSearchConfig(peaks=16),
    )
    candidates, scores, _ = propose(state, dsfw_config, search)
    assert float((candidates["centers"][0] - truth["centers"][0]).norm()) < 1e-4
    assert math.isfinite(float(scores[0]))

"""Spectral Dirichlet atoms against the float64 oracle and their adjoint identity."""

import pytest
import torch
from conftest import close
from reference.oracle import spectral_design

from dsplat import SpectralGrid
from dsplat.kernels import spectral as kernel


@pytest.mark.parametrize(
    "grid",
    [
        SpectralGrid((32,)),
        SpectralGrid((24, 20), window="hann"),
        SpectralGrid((24, 20), samples=(16, 20), window="hamming"),
        SpectralGrid((16, 12), window="kaiser", beta=6.0),
        SpectralGrid((24, 20), radius=4.0),
    ],
)
def test_design_matches_the_oracle(grid):
    centers = torch.rand(7, len(grid.shape), device="cuda") * torch.tensor(
        grid.shape, device="cuda"
    )
    bins = grid.bins()
    a, da = kernel.design(grid, centers, bins, derivatives=True)
    expected, expected_da = spectral_design(grid, centers, bins)
    if grid.radius:  # the truncated kernel is discontinuous at the support boundary
        inside = (expected.abs() > 0)[..., None]
        expected_da, da = expected_da * inside, da * inside.to(da.device)
    close(a, expected)
    close(da, expected_da, 1e-4)


def test_render_jvp_and_vjp_are_one_linear_map(spectral):
    model, params, b = spectral
    grid, centers = model.grid, params["centers"]
    bins = grid.bins()
    a, da = kernel.design(grid, centers, bins, derivatives=True)
    close(kernel.render(grid, centers, b, bins), a @ b)
    dc, db = torch.randn_like(centers), torch.randn_like(b)
    jvp = kernel.jvp(grid, centers, b, dc, db, bins)
    close(jvp, (da * dc[None]).sum(-1) @ b + a @ db, 1e-4)
    upstream = torch.randn(len(bins), dtype=torch.complex64, device="cuda")
    gc, gb = kernel.vjp(grid, centers, b, upstream, bins)
    left = (upstream.conj() * jvp).real.sum()
    right = (gc * dc).sum() + (gb.conj() * db).real.sum()
    torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)
    # The autograd function uses the same backward kernel.
    centers_leaf, b_leaf = centers.clone().requires_grad_(), b.clone().requires_grad_()
    (upstream.conj() * kernel.render(grid, centers_leaf, b_leaf, bins)).real.sum().backward()
    close(centers_leaf.grad, gc, 1e-5)
    close(b_leaf.grad, gb, 1e-5)

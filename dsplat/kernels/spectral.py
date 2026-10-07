"""Bindings of spectral.slang: separable Dirichlet atoms on a spectral grid, in bin units."""

import torch
from torch.autograd.function import once_differentiable

from . import blocks, interleaved, load

WINDOWS = ("rect", "hann", "hamming", "kaiser")


def _args(grid, centers, bins):
    taps = centers.new_zeros((len(grid.shape), max(grid.samples)))
    if grid.window == "kaiser":
        for d, n in enumerate(grid.samples):
            w = torch.kaiser_window(n, periodic=False, beta=grid.beta, device=centers.device)
            taps[d, :n] = w / w.sum()
    device = centers.device
    return dict(
        centers=centers.contiguous(),
        bins=bins.contiguous(),
        samples=torch.tensor(grid.samples, dtype=torch.int32, device=device),
        nfft=torch.tensor(grid.shape, dtype=torch.int32, device=device),
        taps=taps,
        radius=grid.radius,
        window=WINDOWS.index(grid.window),
        dim=len(grid.shape),
        count=len(centers),
        size=len(bins),
    )


def design(grid, centers, bins, derivatives=False):
    """Dictionary A [M, S] and, with derivatives, dA/dcenters [M, S, D]."""
    a = centers.new_empty((len(bins), len(centers), 2))
    da = centers.new_empty((*a.shape[:2], centers.shape[1], 2) if derivatives else (1, 1, 1, 2))
    load("spectral").spectral_design(
        **_args(grid, centers, bins), output=a, jacobian=da, derivatives=derivatives
    ).launchRaw(**blocks(a.numel() // 2))
    a = torch.view_as_complex(a)
    return (a, torch.view_as_complex(da)) if derivatives else a


def _backward(grid, centers, coeff, bins, upstream):
    gc, gb = torch.zeros_like(centers), torch.zeros_like(coeff)
    # 256 bins per block row, more when the grid y limit of 65535 rows would be exceeded.
    span = 256 * -(-len(bins) // (256 * 65535))
    load("spectral").spectral_backward(
        **_args(grid, centers, bins), coeff=coeff, upstream=upstream, grad_centers=gc,
        grad_coeff=gb, span=span,
    ).launchRaw(blockSize=(64, 1, 1), gridSize=(-(-len(centers) // 64), -(-len(bins) // span), 1))  # fmt: skip
    return gc, gb


class _Render(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, coeff, grid, bins):
        output = centers.new_empty((len(bins), 2))
        load("spectral").spectral_forward(
            **_args(grid, centers, bins), coeff=coeff, output=output
        ).launchRaw(**blocks(len(bins), 128))
        ctx.save_for_backward(centers, coeff, bins)
        ctx.grid = grid
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        centers, coeff, bins = ctx.saved_tensors
        return (*_backward(ctx.grid, centers, coeff, bins, grad.contiguous()), None, None)


def render(grid, centers, coefficients, bins):
    """Coherent sum of complex atoms at the given bins; differentiable in both inputs."""
    output = _Render.apply(centers.contiguous(), interleaved(coefficients), grid, bins)
    return torch.view_as_complex(output)


def jvp(grid, centers, coefficients, delta_centers, delta_coefficients, bins):
    output = centers.new_empty((len(bins), 2))
    load("spectral").spectral_jvp(
        **_args(grid, centers, bins),
        coeff=interleaved(coefficients),
        delta_centers=delta_centers.contiguous(),
        delta_coeff=interleaved(delta_coefficients),
        output=output,
    ).launchRaw(**blocks(len(bins), 128))
    return torch.view_as_complex(output)


def vjp(grid, centers, coefficients, upstream, bins):
    """Gradients of Re(conj(upstream) . render) for centers and complex coefficients."""
    gc, gb = _backward(
        grid, centers.contiguous(), interleaved(coefficients), bins, interleaved(upstream)
    )
    return gc, torch.view_as_complex(gb)

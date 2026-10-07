"""Dirichlet atoms on a spectral grid: the signal-level measurement model.

An atom has a center [D] in bin units and a complex amplitude (its coefficient).
Its spectrum is the separable windowed Dirichlet kernel of every axis, observed at
every grid bin. This is the representation of the 1D/2D signal experiments.
"""

from dataclasses import dataclass
from math import prod

import torch

from .kernels import spectral as kernel


@dataclass(frozen=True)
class SpectralGrid:
    shape: tuple[int, ...]  # FFT length per axis, one to three axes
    samples: tuple[int, ...] | None = None  # observed samples per axis; None: the FFT length
    window: str = "rect"  # rect, hann, hamming or kaiser
    beta: float = 14.0  # Kaiser shape
    radius: float = 0.0  # kernel support in bins; zero is the exact, untruncated response

    def __post_init__(self):
        if self.samples is None:
            object.__setattr__(self, "samples", self.shape)

    @property
    def size(self):
        return prod(self.shape)

    def bins(self, device="cuda"):
        axes = [torch.arange(n, device=device, dtype=torch.float32) for n in self.shape]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))


@dataclass(frozen=True)
class SpectralModel:
    """The measurement model of atoms with centers [S, D] on one grid.

    It offers the operations the solver uses: render, jvp, vjp, design and jacobian
    on chosen rows, streamed least-squares blocks, correlations and row sampling.
    It has no apertures, so the aperture-subset options never apply.
    """

    grid: SpectralGrid
    row_alignment = 1
    apertures = 0

    @property
    def shape(self):
        return self.grid.shape

    @property
    def size(self):
        return self.grid.size

    def prepare(self, specs):
        """Nothing to precompute: every grid bin is observed."""
        return self

    def design(self, params, rows):
        centers = params["centers"]
        return kernel.design(self.grid, centers, self.grid.bins(centers.device)[rows])

    def jacobian(self, params, rows, atoms, keys):
        centers = params["centers"]
        bins = self.grid.bins(centers.device)[rows]
        return {"centers": kernel.design(self.grid, centers[atoms], bins, derivatives=True)[1]}

    def render(self, params, coefficients):
        centers = params["centers"]
        bins = self.grid.bins(centers.device)
        return kernel.render(self.grid, centers, coefficients, bins).reshape(self.shape)

    def jvp(self, params, coefficients, tangent, delta_coefficients):
        centers = params["centers"]
        bins = self.grid.bins(centers.device)
        value = kernel.jvp(
            self.grid, centers, coefficients, tangent["centers"], delta_coefficients, bins
        )
        return value.reshape(self.shape)

    def vjp(self, params, coefficients, upstream):
        centers = params["centers"]
        bins = self.grid.bins(centers.device)
        gc, gb = kernel.vjp(self.grid, centers, coefficients, upstream.flatten(), bins)
        return {"centers": gc}, gb

    def correlate(self, params, residual):
        """a^H r and |a|^2 for every atom; an FFT when the atoms are exactly the grid bins."""
        centers = params["centers"]
        bins = self.grid.bins(centers.device)
        if centers.shape == bins.shape and torch.equal(centers, bins):
            atom = kernel.design(self.grid, torch.zeros_like(centers[:1]), bins).reshape(self.shape)
            spectrum = torch.fft.fftn(residual.reshape(self.shape)) * torch.fft.fftn(atom).conj()
            norm = atom.abs().square().sum().expand(len(centers))
            return torch.fft.ifftn(spectrum).flatten(), norm
        corr = residual.new_zeros(len(centers))
        norm = corr.real.clone()
        chunk = max(1, 2**24 // len(centers))  # bounds the streamed dictionary block
        for start in range(0, self.size, chunk):
            a = self.design(params, slice(start, start + chunk))
            corr += a.H @ residual.flatten()[start : start + chunk]
            norm += a.abs().square().sum(0)
        return corr, norm

    def blocks(self, params, target, real, chunk):
        flat = target.flatten()
        for start in range(0, self.size, chunk):
            rows = slice(start, start + chunk)
            a, y = self.design(params, rows), flat[rows]
            if real:
                a = torch.view_as_real(a).permute(0, 2, 1).reshape(-1, a.shape[1])
                y = torch.view_as_real(y).reshape(-1)
            yield a, y

    def sample_rows(self, count, seed, device):
        # Deterministic broad coverage; all rows when count reaches the grid size.
        return torch.linspace(0, self.size - 1, min(count, self.size), device=device).long()

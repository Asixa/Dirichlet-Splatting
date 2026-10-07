"""FMCW sensor: chirp configuration, scans, and surfels measured as range profiles.

A linear chirp of bandwidth B and carrier fc is sampled by adc_samples ADC samples
and zero-padded to n_fft. One scan holds the positive range bins of every aperture,
data[A, K] = FFT(ADC, n_fft)[:K] / adc_samples. A surfel at range r appears at range
bin r / spacing, spacing = c / (2 B pad_factor), as the Dirichlet kernel of the
range FFT times the chirp phase; surfel.py has its amplitude.

FMCWModel is the measurement model the solver works with. Besides rendering and
derivatives it projects streamed rows onto the row basis of the range response,
draws seeded aperture subsets, and provides the range gate, the FFT range table and
the normal-bank scores used by the spatial search. migration is an FFT
angular-spectrum initializer for planar rasters.
"""

import math
from dataclasses import dataclass, field, replace
from math import ceil, floor
from typing import NamedTuple

import torch
import torch.nn.functional as F

from . import surfel
from .kernels import search
from .kernels.response import range_response


@dataclass(frozen=True)
class FMCWConfig:
    fc_hz: float = 120e9
    bandwidth_hz: float = 15e9
    adc_samples: int = 128
    sweep_time_s: float = 1e-5
    adc_start_time_s: float = 0.0
    pad_factor: int = 2
    c0_m_s: float = 299792458.0

    @property
    def lambda_m(self):
        return self.c0_m_s / self.fc_hz

    @property
    def slope_hz_s(self):
        return self.bandwidth_hz / self.sweep_time_s

    @property
    def sample_rate_hz(self):
        return self.adc_samples / self.sweep_time_s

    @property
    def n_fft(self):
        return self.adc_samples * self.pad_factor

    @property
    def num_range_bins(self):
        return self.n_fft // 2

    @property
    def range_bin_spacing_m(self):
        return self.c0_m_s / (2 * self.bandwidth_hz * self.pad_factor)

    @property
    def range_resolution_m(self):
        return self.c0_m_s / (2 * self.bandwidth_hz)


@dataclass(frozen=True)
class Scan:
    """Range profiles data [A, K] measured at scan positions [A, 3], metres."""

    data: torch.Tensor
    scan_positions_m: torch.Tensor
    fmcw_config: FMCWConfig


# --- Row basis of the range response ---------------------------------------------------------

PEAKS_PER_BIN = 16  # samples of the peak position per range bin for the response SVD


@dataclass(frozen=True)
class RowBasis:
    """Orthonormal row projection for observations stored as consecutive aperture blocks."""

    adjoint: torch.Tensor  # complex [rank, bins]; U^H
    interleaved: torch.Tensor  # real [2 rank, 2 bins] acting on (re, im) row pairs

    def reduce(self, values):
        """Complex rows [blocks * bins, ...] to [blocks * rank, ...]."""
        tail = values.shape[1:]
        rows = values.reshape(-1, self.adjoint.shape[1], tail.numel())
        return (self.adjoint @ rows).reshape(-1, *tail)

    def reduce_major(self, rows):
        """Real rows ordered bin by bin, (2 k + part) * blocks + block, in one product."""
        tail = rows.shape[1:]
        return (self.interleaved @ rows.reshape(self.interleaved.shape[1], -1)).reshape(-1, *tail)


def response_basis(config, low, high, device):
    """Row basis of every peak position in [low, high] range bins, or None if it spans all bins.

    Singular values below eps / 128 of the largest are dropped, so the normal
    equations are preserved to float32 precision.
    """
    count = max(2 * config.adc_samples, int(PEAKS_PER_BIN * (high - low)) + 1)
    # The SVD stays on the host in float64 to keep the tight rank cutoff.
    response = range_response(config, low, high, count, device).cpu()
    vectors, values, _ = torch.linalg.svd(response, full_matrices=False)
    rank = int((values > torch.finfo(torch.float32).eps / 128 * values[0]).sum())
    if rank >= config.num_range_bins:
        return None
    adjoint = vectors[:, :rank].mH
    interleaved = torch.zeros((2 * rank, 2 * config.num_range_bins), dtype=torch.float64)
    interleaved[0::2, 0::2], interleaved[0::2, 1::2] = adjoint.real, -adjoint.imag
    interleaved[1::2, 0::2], interleaved[1::2, 1::2] = adjoint.imag, adjoint.real
    return RowBasis(adjoint.to(device, torch.complex64), interleaved.to(device, torch.float32))


class RangeTable(NamedTuple):
    """FFT-interpolated range correlations of a residual; they only rank proposals."""

    correlation: torch.Tensor  # [apertures, n_fft * oversample, 2]
    norm: torch.Tensor  # [n_fft * oversample]; keeps the observed-bin mask
    oversample: int


# --- Measurement model -----------------------------------------------------------------------


@dataclass(frozen=True)
class FMCWModel:
    """Surfels measured by an FMCW scan: the measurement model of the physical experiments.

    params = {centers [S, 3], normals [S, 3], areas [S]} (see surfel.py); the
    coefficients are reflectivities. It offers the operations the solver uses:
    render, jvp, vjp, design and jacobian on chosen rows, streamed least-squares
    blocks, correlations, and seeded row and aperture sampling. prepare returns a
    copy with the row basis; block_bases only memoizes bases derived from it.
    """

    scan: Scan
    fresnel_n: float = 0.0  # known real refractive index of the surfels; <= 1 for none
    basis: RowBasis | None = None  # row projection of every center in the box; see prepare
    # Per aperture, on the host: lowest and highest peak bin of any center in the box.
    bands: tuple | None = None
    # Bases already built by block_basis, keyed by their band of range bins.
    block_bases: dict = field(default_factory=dict, repr=False)

    @property
    def shape(self):
        return tuple(self.scan.data.shape)

    @property
    def size(self):
        return self.scan.data.numel()

    @property
    def apertures(self):
        return self.shape[0]

    @property
    def row_alignment(self):
        """Streamed blocks hold whole aperture profiles."""
        return self.shape[1]

    @property
    def sensor(self):
        """The chirp configuration and the sensor positions [A, 3]."""
        return self.scan.fmcw_config, self.scan.scan_positions_m

    def sample_apertures(self, count, seed):
        """Sorted ids of a seeded random set of whole apertures, and the model on them.

        The subset keeps the sensor, material and shared row basis, but not the
        per-aperture bands. The same seed gives the same set as sample_rows with
        count * bins rows.
        """
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randperm(self.shape[0], generator=generator)[: max(1, count)]
        ids = ids.sort().values.to(self.scan.data.device)
        scan = replace(
            self.scan,
            data=self.scan.data[ids].contiguous(),
            scan_positions_m=self.scan.scan_positions_m[ids].contiguous(),
        )
        return ids, replace(self, scan=scan, bands=None, block_bases={})

    def sample_rows(self, count, seed, device):
        generator = torch.Generator().manual_seed(seed)
        bins = self.shape[1]
        apertures = torch.randperm(self.shape[0], generator=generator)[: max(1, count // bins)]
        return (apertures.to(device)[:, None] * bins + torch.arange(bins, device=device)).flatten()

    def prepare(self, specs):
        """This model with the row basis of every center inside the box of the centers' bounds.

        The range-bin band follows from the nearest and farthest box points seen by
        any aperture, with one bin of margin. Without a finite box, or when the band
        wraps around the FFT, the basis covers every peak position.
        """
        lower, upper = specs["centers"].lower, specs["centers"].upper
        config, positions = self.sensor
        low, high, bands = 0.0, float(config.n_fft), None
        if lower is not None and upper is not None:
            lo, hi = positions.new_tensor(lower).expand(3), positions.new_tensor(upper).expand(3)
            nearest = torch.minimum(torch.maximum(positions, lo), hi)
            farthest = torch.where((positions - lo).abs() > (positions - hi).abs(), lo, hi)
            spacing = config.range_bin_spacing_m
            nearest = ((positions - nearest).norm(dim=1) / spacing - 1).clamp(min=0)
            farthest = (positions - farthest).norm(dim=1) / spacing + 1
            near, far = float(nearest.min()), float(farthest.max())
            if far - near < config.n_fft:
                low, high = near, far
                bands = (nearest.cpu().numpy(), farthest.cpu().numpy())
        basis = response_basis(config, low, high, positions.device)
        return replace(self, basis=basis, bands=bands, block_bases={})

    def block_basis(self, rows):
        """Row basis for a block of whole apertures, at most as large as the shared one.

        A block uses the union of its apertures' own range bands, widened to whole
        critical range samples so that few distinct bases arise. They are built on
        first use from host-side bands, so the stream never waits for the device.
        """
        if self.bands is None or self.basis is None:
            return self.basis
        bins, step = self.shape[1], self.scan.fmcw_config.pad_factor
        first, last = rows.start // bins, -(-rows.stop // bins)
        key = (
            floor(self.bands[0][first:last].min() / step) * step,
            ceil(self.bands[1][first:last].max() / step) * step,
        )
        if key not in self.block_bases:
            config, positions = self.sensor
            basis = response_basis(config, float(key[0]), float(key[1]), positions.device)
            # Rounding the band outward can leave nothing to gain over the shared basis.
            smaller = basis is not None and len(basis.adjoint) < len(self.basis.adjoint)
            self.block_bases[key] = basis if smaller else self.basis
        return self.block_bases[key]

    def blocks(self, params, target, real, chunk):
        """Least-squares blocks of whole apertures, projected onto the row basis.

        Real blocks are ordered bin by bin so that one product projects them.
        """
        config, positions = self.sensor
        bins = self.shape[1]
        flat = target.flatten()
        for start in range(0, self.size, chunk):
            rows = slice(start, min(start + chunk, self.size))
            basis = self.block_basis(rows)
            if real:
                block = positions[rows.start // bins : rows.stop // bins]
                a = surfel.design(config, block, params, self.fresnel_n, real=True)
                y = torch.view_as_real(flat[rows]).reshape(len(block), 2 * bins).T.reshape(-1)
                yield (a, y) if basis is None else (basis.reduce_major(a), basis.reduce_major(y))
            else:
                a, y = self.design(params, rows), flat[rows]
                yield (a, y) if basis is None else (basis.reduce(a), basis.reduce(y))

    def _apertures(self, rows):
        """Apertures of arbitrary row ids, each row's aperture among them, and its bin."""
        apertures, inverse = torch.unique(rows // self.shape[1], return_inverse=True)
        return self.scan.scan_positions_m[apertures], inverse, rows % self.shape[1]

    def design(self, params, rows):
        """Complex dictionary rows [rows, S]; a slice must cover whole apertures."""
        config, positions = self.sensor
        bins = self.shape[1]
        if isinstance(rows, slice):
            block = positions[rows.start // bins : -(-rows.stop // bins)]
            return surfel.design(config, block, params, self.fresnel_n)
        block, inverse, ids = self._apertures(rows)
        a = surfel.design(config, block, params, self.fresnel_n)
        return a.reshape(len(block), bins, -1)[inverse, ids]

    def jacobian(self, params, rows, atoms, keys):
        """Column derivatives [rows, atoms, D] for the given parameter keys and row ids."""
        block, inverse, ids = self._apertures(rows)
        selected = {key: value[atoms] for key, value in params.items()}
        derivatives = surfel.jacobian(self.scan.fmcw_config, block, selected, self.fresnel_n)
        return {key: derivatives[key][inverse, ids] for key in keys}

    def render(self, params, coefficients):
        return surfel.render(*self.sensor, params, coefficients, self.fresnel_n)

    def jvp(self, params, coefficients, tangent, delta_coefficients):
        return surfel.jvp(
            *self.sensor, params, coefficients, tangent, delta_coefficients, self.fresnel_n
        )

    def vjp(self, params, coefficients, upstream):
        return surfel.vjp(*self.sensor, params, coefficients, upstream, self.fresnel_n)

    def correlate(self, params, residual):
        return surfel.correlate(*self.sensor, params, residual.reshape(self.shape), self.fresnel_n)

    def correlate_gradient(self, params, residual, include_normals=False):
        residual = residual.reshape(self.shape)
        return surfel.correlate_gradient(
            *self.sensor, params, residual, self.fresnel_n, include_normals
        )

    def correlate_orientations(self, params, residual, bank, table=None):
        residual = residual.reshape(self.shape)
        return surfel.correlate_orientations(
            *self.sensor, params, bank, residual, self.fresnel_n, table
        )

    @torch.no_grad()
    def range_gate(self, points, measurement, relative_power, margin_cells, min_fraction):
        """Points whose spherical range falls in a measured range interval at enough apertures.

        Each of 32 evenly spaced apertures thresholds its profile in measurement
        relative to its own peak, keeps every interval above it and dilates it by
        margin_cells range resolutions. A point must hit an interval at min_fraction
        of the apertures with signal. This filters proposals: weak or hidden
        reflectors can be dropped.
        """
        config, positions = self.sensor
        count = min(32, len(measurement))
        ids = torch.linspace(0, len(measurement) - 1, count, device=points.device).long()
        power = measurement[ids].abs().square()
        peak = power.max(-1).values
        informative = peak > peak.max() * 1e-6
        power, peak = power[informative], peak[informative]
        spacing = config.range_bin_spacing_m
        radius = math.ceil(margin_cells * config.range_resolution_m / spacing)
        above = (power >= relative_power * peak[:, None]).float()[:, None]
        intervals = F.max_pool1d(above, 2 * radius + 1, stride=1, padding=radius)[:, 0].int()
        return search.range_gate(
            points, positions[ids][informative], intervals, spacing, min_fraction
        )

    @torch.no_grad()
    def range_table(self, residual, oversample=8):
        """Oversampled FFT range correlations of the residual for proposal scores.

        Padding the inverse FFT samples the off-grid correlations densely.
        """
        config = self.scan.fmcw_config
        n, samples = config.n_fft, config.adc_samples
        size = n * oversample
        residual = residual.reshape(self.shape)
        coefficients = torch.fft.ifft(residual, n=n, dim=-1)[..., :samples] * (n / samples)
        correlation = torch.fft.fft(coefficients, n=size, dim=-1)
        kernel = torch.fft.fft(residual.real.new_ones(samples), n=size) / samples
        mask = residual.real.new_zeros(size)
        mask[: config.num_range_bins * oversample : oversample] = 1
        norm = torch.fft.ifft(torch.fft.fft(mask) * torch.fft.fft(kernel.abs().square())).real
        return RangeTable(
            torch.view_as_real(correlation).contiguous(), norm.contiguous(), oversample
        )


# --- Initialization --------------------------------------------------------------------------


@torch.no_grad()
def migration(scan, points, aperture_shape, *, depth_samples=96, padding=2, frequency_chunk=16):
    """Focus a uniformly sampled planar aperture at world-space candidate points.

    Returns relative complex focusing strengths, not calibrated reflectivities. The
    scan must be a complete row-major planar raster, and the points must lie on one
    side of it. Positive range bins are zero-filled and RVP-corrected to approximate
    chirp samples. Angular-spectrum propagation uses K = 4 pi f / c and evaluates
    every propagating spatial-frequency replica of an undersampled aperture. The
    scalar Green model omits normal and Fresnel factors: this is an initializer.
    """
    nu, nv = aperture_shape
    positions = scan.scan_positions_m.reshape(nu, nv, 3)
    origin = positions[nu // 2, nv // 2]
    u = (positions[-1, 0] - positions[0, 0]) / (nu - 1)
    v = (positions[0, -1] - positions[0, 0]) / (nv - 1)
    du, dv = float(u.norm()), float(v.norm())
    u, v = u / du, v / dv
    local = (points - origin) @ torch.stack([u, v, torch.linalg.cross(u, v)], -1)
    local[:, 2] = local[:, 2].abs()
    zmin, zmax = local[:, 2].min(), local[:, 2].max()
    if float(zmax - zmin) < 1e-6:
        depth_samples = 1
    cfg = scan.fmcw_config
    device, dtype = points.device, points.dtype
    beat = (
        torch.arange(cfg.num_range_bins, device=device, dtype=dtype)
        * cfg.sample_rate_hz
        / cfg.n_fft
    )
    corrected = scan.data * torch.exp(1j * math.pi * beat.square() / cfg.slope_hz_s)
    adc = torch.fft.ifft(corrected, n=cfg.n_fft, dim=-1)[..., : cfg.adc_samples] * cfg.adc_samples
    pu, pv = nu * padding, nv * padding
    padded = adc.new_zeros((cfg.adc_samples, pu, pv))
    su, sv = pu // 2 - nu // 2, pv // 2 - nv // 2
    padded[:, su : su + nu, sv : sv + nv] = adc.reshape(nu, nv, -1).permute(2, 0, 1)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(padded, dim=(-2, -1))), dim=(-2, -1)
    )
    time = (
        cfg.adc_start_time_s
        + torch.arange(cfg.adc_samples, device=device, dtype=dtype) / cfg.sample_rate_hz
    )
    k = 4 * math.pi * (cfg.fc_hz + cfg.slope_hz_s * time) / cfg.c0_m_s
    carrier = k.max() / 2
    distinct = torch.unique(local[:, 2])
    regular = len(distinct) < 3 or torch.allclose(
        distinct.diff(), distinct.diff().mean().expand(len(distinct) - 1), atol=1e-7, rtol=1e-4
    )
    if len(distinct) <= depth_samples and regular:
        depths = distinct  # queries already lie on these regular planes
    else:
        # |kz - carrier| <= Kmax / 2: four samples per Nyquist interval bound the
        # interpolation error without assuming paraxial waves.
        count = max(depth_samples, 1 + math.ceil(float(zmax - zmin) * float(k.max()) * 2 / math.pi))
        depths = torch.linspace(float(zmin), float(zmax), count, device=device, dtype=dtype)
    # A sampled aperture spectrum is periodic; keep every propagating replica.
    copies_u = max(1, math.ceil(float(k.max()) * du / math.pi))
    copies_v = max(1, math.ceil(float(k.max()) * dv / math.pi))
    qu, qv = pu * copies_u, pv * copies_v
    iu = (torch.arange(qu, device=device) - qu // 2 + pu // 2) % pu
    iv = (torch.arange(qv, device=device) - qv // 2 + pv // 2) % pv
    dx, dy = du / copies_u, dv / copies_v
    kx = 2 * math.pi * torch.fft.fftshift(torch.fft.fftfreq(qu, d=dx, device=device, dtype=dtype))
    ky = 2 * math.pi * torch.fft.fftshift(torch.fft.fftfreq(qv, d=dy, device=device, dtype=dtype))
    transverse = kx[:, None].square() + ky[None].square()
    spacing = local.new_tensor([dx, dy])
    low = torch.floor(local[:, :2].min(0).values / spacing).long() - 2
    high = torch.ceil(local[:, :2].max(0).values / spacing).long() + 2
    a, b = int(low[0]) + qu // 2, int(high[0]) + qu // 2 + 1
    c, d = int(low[1]) + qv // 2, int(high[1]) + qv // 2 + 1
    slices = []
    for depth in depths:
        focused = spectrum.new_zeros((qu, qv))
        for start in range(0, cfg.adc_samples, frequency_chunk):
            part = slice(start, start + frequency_chunk)
            kz = (k[part, None, None].square() - transverse).clamp(min=0).sqrt()
            focused += (spectrum[part][:, iu][:, :, iv] * kz * torch.exp(-1j * kz * depth)).sum(0)
        plane = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(focused)))
        # Shift the full propagating band before interpolation, then restore it.
        slices.append(plane[a:b, c:d] * torch.exp(1j * carrier * depth))
    volume = torch.stack(slices) * (copies_u * copies_v / cfg.adc_samples)
    xy = 2 * (local[:, :2] - low * spacing) / ((high - low) * spacing) - 1
    zz = (
        2 * (local[:, 2] - zmin) / (zmax - zmin) - 1
        if len(depths) > 1
        else torch.zeros_like(local[:, 2])
    )
    coordinates = torch.stack([xy[:, 1], xy[:, 0], zz], -1).reshape(1, 1, 1, -1, 3)
    channels = torch.view_as_real(volume).permute(3, 0, 1, 2)[None]
    sampled = F.grid_sample(channels, coordinates, mode="bilinear", align_corners=True).reshape(
        2, -1
    )
    return torch.complex(sampled[0], sampled[1]) * torch.exp(-1j * carrier * local[:, 2])

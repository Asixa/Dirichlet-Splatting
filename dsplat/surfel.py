"""Oriented surfels: the scattering primitive of the physical measurements.

A surfel is a small planar reflector. Its parameters, one row per surfel:

    centers  [S, 3]  position in metres
    normals  [S, 3]  outward unit normal
    areas    [S]     area in square metres

Its reflectivity b is the linear coefficient that variable projection solves, and
its material is a known real refractive index fresnel_n (<= 1 for none). Seen from a
monostatic sensor at p, with r = |p - c|, u = (p - c) / r and cos = u . n:

    amplitude = sqrt(area) * cos^2 * F(cos) * lambda / (8 pi r)   zero when cos <= 0
    F(cos)    = (root - cos) / (root + cos),  root = sqrt(fresnel_n^2 - 1 + cos^2)
    delay     = 2 r / c

The echo depends on area and reflectivity only through sqrt(area) * b, so fitting
both is degenerate: keep the area fixed (ParameterConfig(trainable=False)) and read
the amplitude from b. A complex b also absorbs the carrier phase 4 pi r / lambda,
which otherwise pins r to a fraction of the wavelength; with a real b the range is
ambiguous only by a sign, lambda / 4.

F is the s-polarized Fresnel reflection of the dielectric. A chirped sensor turns
amplitude and delay into an echo with phase 2 pi (fc tau + K tau (t0 - tau / 2)) at
range bin r / spacing (fmcw.py). The Slang kernels evaluate these formulas, written
once in kernels/scattering.slang. The functions below take the chirp as an
FMCWConfig config, the sensor positions [A, 3] and the surfel parameters, and return range
profiles [A, K], dictionary columns or correlations.

render_scene is the paper's general 13-DOF surfel composition instead: position,
unit normal, area, complex reflectivity, velocity and complex impedance, rendered
in range, Doppler and azimuth bins.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kernels import spectral as spectral_kernel
from .kernels import surfel as kernel
from .spectral import SpectralGrid


def render(config, positions, params, coefficients, fresnel_n=0.0):
    """Range profiles [A, K] of surfels with reflectivities coefficients; differentiable."""
    weights = coefficients * params["areas"].sqrt()
    return kernel.render(
        config, positions, params["centers"], params["normals"], weights, fresnel_n
    )


def jvp(config, positions, params, coefficients, tangent, delta_coefficients, fresnel_n=0.0):
    """Change of the profiles along tangent (centers, normals, areas) and delta_coefficients."""
    scale = params["areas"].sqrt()
    weights = coefficients * scale
    delta_weights = delta_coefficients * scale + coefficients * tangent["areas"] / (2 * scale)
    return kernel.jvp(
        config, positions, params["centers"], params["normals"], weights,
        tangent["centers"], tangent["normals"], delta_weights, fresnel_n,
    )  # fmt: skip


def vjp(config, positions, params, coefficients, upstream, fresnel_n=0.0):
    """Gradients of Re(conj(upstream) . profiles): per parameter, and for the coefficients."""
    scale = params["areas"].sqrt()
    centers, normals, weights = kernel.vjp(
        config,
        positions,
        params["centers"],
        params["normals"],
        coefficients * scale,
        upstream,
        fresnel_n,
    )
    areas = (weights.conj() * coefficients).real / (2 * scale)
    return dict(centers=centers, normals=normals, areas=areas), weights * scale


def design(config, positions, params, fresnel_n=0.0, real=False):
    """Dictionary columns of every surfel for whole apertures; see kernels.surfel.design."""
    return kernel.design(
        config, positions, params["centers"], params["normals"], params["areas"], fresnel_n, real
    )


def jacobian(config, positions, params, fresnel_n=0.0):
    """Derivatives of the columns [A, K, S] by centers and normals [..., 3] and areas [..., 1]."""
    columns, dc, dn = kernel.jacobian(
        config, positions, params["centers"], params["normals"], fresnel_n
    )
    scale = params["areas"].sqrt()
    return {
        "centers": dc * scale[:, None],
        "normals": dn * scale[:, None],
        "areas": (columns / (2 * scale))[..., None],
    }


def correlate(config, positions, params, residual, fresnel_n=0.0):
    """a^H r and |a|^2 of every surfel's column a against the residual profiles [A, K]."""
    corr, norm = kernel.correlate(
        config, positions, params["centers"], params["normals"], residual, fresnel_n
    )
    return corr * params["areas"].sqrt(), norm * params["areas"]


def correlate_gradient(config, positions, params, residual, fresnel_n=0.0, include_normals=False):
    """correlate and its derivatives by centers (and normals): corr, norm, dcorr, dnorm."""
    corr, norm, dcorr, dnorm = kernel.correlate_gradient(
        config,
        positions,
        params["centers"],
        params["normals"],
        residual,
        fresnel_n,
        include_normals,
    )
    scale, area = params["areas"].sqrt(), params["areas"]
    return corr * scale, norm * area, dcorr * scale[:, None], dnorm * area[:, None]


def correlate_orientations(config, positions, params, bank, residual, fresnel_n=0.0, table=None):
    """correlate for each surfel's own normal and every bank normal, [S, 1 + bank]."""
    corr, norm = kernel.correlate_orientations(
        config, positions, params["centers"], params["normals"], bank, residual, fresnel_n, table
    )
    scale = params["areas"].sqrt()[:, None]
    return corr * scale, norm * scale.square()


# --- General 13-DOF composition --------------------------------------------------------------


@dataclass(frozen=True)
class Sensor:
    """A range/Doppler/azimuth sensor: bins per axis and the chirp constants."""

    shape: tuple[int, int, int] = (128, 32, 16)  # range, Doppler, azimuth
    carrier_hz: float = 77e9
    sample_rate_hz: float = 2e6
    slope_hz_s: float = 20e12
    chirp_interval_s: float = 60e-6
    spacing_m: float | None = None  # array pitch; None: half a wavelength
    wave_speed: float = 299792458.0

    @property
    def wavelength(self):
        return self.wave_speed / self.carrier_hz


def project(positions, velocities, sensor: Sensor, origin=None, array_axis=None):
    """Range, Doppler and azimuth bin coordinates of moving points."""
    origin = positions.new_zeros(3) if origin is None else origin
    array_axis = positions.new_tensor([1, 0, 0]) if array_axis is None else array_axis
    delta = positions - origin
    r = delta.norm(dim=-1).clamp(min=1e-9)
    direction = delta / r[:, None]
    nr, nd, na = sensor.shape
    spacing = sensor.wavelength / 2 if sensor.spacing_m is None else sensor.spacing_m
    mu_r = nr / sensor.sample_rate_hz * 2 * sensor.slope_hz_s / sensor.wave_speed * r
    mu_d = 2 * nd * sensor.chirp_interval_s / sensor.wavelength * (velocities * direction).sum(-1)
    mu_a = na * spacing / sensor.wavelength * (direction * array_axis).sum(-1)
    return torch.stack([mu_r, mu_d, mu_a], -1)


def fresnel_te(cosine, impedance, ambient=1.0):
    """Complex TE reflection of a material with complex impedance; 0 impedance gives -1."""
    relative = impedance / ambient
    # At zero cosine the geometric amplitude is already zero. Evaluate Fresnel at a
    # benign cosine there to avoid 0*NaN at a matched interface.
    safe_cosine = torch.where(cosine > 0, cosine, torch.ones_like(cosine))
    transmitted = torch.sqrt(1 - relative.square() * (1 - safe_cosine.square()) + 0j)
    coefficient = (relative * safe_cosine - transmitted) / (relative * safe_cosine + transmitted)
    return torch.where(cosine > 0, coefficient, torch.where(relative == 1, 0j, -1 + 0j))


def render_scene(
    positions, normals, areas, reflectivity, velocities, impedance, sensor: Sensor,
    *, origin=None, gamma=1.0, window="rect",
):  # fmt: skip
    """Coherent range/Doppler/azimuth cube of 13-DOF surfels; differentiable in all of them.

    The coefficient is reflectivity * sqrt(area) * cos^2 * F_TE(cos) * r^-gamma with the
    round-trip carrier phase; the per-axis response is the Slang spectral kernel.
    """
    origin = positions.new_zeros(3) if origin is None else origin
    delta = origin - positions
    r = delta.norm(dim=-1).clamp(min=1e-9)
    cos = (F.normalize(normals, dim=-1) * delta / r[:, None]).sum(-1).clamp(min=0)
    coeff = reflectivity * areas.sqrt() * cos.square() * fresnel_te(cos, impedance) * r.pow(-gamma)
    coeff = coeff * torch.exp(4j * torch.pi * r / sensor.wavelength)
    grid = SpectralGrid(sensor.shape, window=window)
    centers = project(positions, velocities, sensor, origin)
    bins = grid.bins(positions.device)
    return spectral_kernel.render(grid, centers, coeff, bins).reshape(grid.shape)

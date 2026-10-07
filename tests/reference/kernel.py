"""Independent PyTorch reference for the unit-peak, negative-exponent DFT PSF."""

import math

import torch


def dirichlet(delta: torch.Tensor, samples: int, n_fft: int | None = None):
    """Return d and d/d(delta), including removable singularities and zero padding.

    d(delta) = mean_n exp(-2 pi i n delta / n_fft), n=0,...,samples-1.
    The stable ratio uses a Taylor expansion; quotient derivatives never divide by d.
    """
    n_fft = samples if n_fft is None else n_fft
    x = delta - n_fft * torch.round(delta / n_fft)
    a, b = math.pi * samples / n_fft, math.pi / n_fft
    h = math.pi * (samples - 1) / n_fft
    near = x.abs() < 1e-3
    safe = torch.where(near, torch.ones_like(x), x)
    sa, sb = torch.sin(a * safe), torch.sin(b * safe)
    ratio = sa / (samples * sb)
    deriv = (a * torch.cos(a * safe) * sb - b * sa * torch.cos(b * safe)) / (samples * sb.square())
    c2 = (b * b - a * a) / 6
    c4 = (3 * a**4 + 7 * b**4 - 10 * a * a * b * b) / 360
    ratio = torch.where(near, 1 + c2 * x.square() + c4 * x**4, ratio)
    deriv = torch.where(near, 2 * c2 * x + 4 * c4 * x**3, deriv)
    phase = torch.exp(-1j * h * x)
    return phase * ratio, phase * (deriv - 1j * h * ratio)


def windowed_kernel(delta, samples, n_fft=None, window="rect", beta=14.0):
    """Periodic Hann/Hamming windows normalized by their sum, or rectangular."""
    n_fft = samples if n_fft is None else n_fft
    if window == "rect":
        return dirichlet(delta, samples, n_fft)
    if window == "kaiser":
        delta = delta - n_fft * torch.round(delta / n_fft)
        taps = torch.kaiser_window(
            samples, periodic=False, beta=beta, device=delta.device, dtype=delta.dtype
        )
        taps = taps / taps.sum()
        value = torch.zeros_like(
            delta, dtype=torch.complex128 if delta.dtype == torch.float64 else torch.complex64
        )
        derivative = torch.zeros_like(value)
        for n in range(samples):
            factor = -2j * math.pi * n / n_fft
            term = taps[n] * torch.exp(factor * delta)
            value = value + term
            derivative = derivative + factor * term
        return value, derivative
    a, b = (0.5, 0.5) if window == "hann" else (0.54, 0.46)
    shift = n_fft / samples
    k, dk = dirichlet(delta, samples, n_fft)
    km, dkm = dirichlet(delta - shift, samples, n_fft)
    kp, dkp = dirichlet(delta + shift, samples, n_fft)
    return k - b / (2 * a) * (km + kp), dk - b / (2 * a) * (dkm + dkp)

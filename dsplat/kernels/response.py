"""Binding of response.slang."""

import torch

from . import blocks, load


def range_response(config, low, high, count, device):
    """Float64 range responses [bins, count] of count peak positions spread over [low, high]."""
    output = torch.empty((config.num_range_bins, count, 2), device=device, dtype=torch.float64)
    load("response").range_response(
        output=output,
        low=low,
        high=high,
        count=count,
        bins=config.num_range_bins,
        samples=config.adc_samples,
        nfft=config.n_fft,
    ).launchRaw(**blocks(config.num_range_bins * count))
    return torch.view_as_complex(output)

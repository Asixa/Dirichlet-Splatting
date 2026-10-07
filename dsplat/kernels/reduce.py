"""Bindings of reduce.slang."""

import torch

from . import blocks, load


def energy(value, reference):
    """sum |value - reference|^2 of two complex tensors in one reduction."""
    value = torch.view_as_real(value.resolve_conj().resolve_neg().contiguous()).reshape(-1, 2)
    reference = torch.view_as_real(reference.resolve_conj().resolve_neg().contiguous()).reshape(
        -1, 2
    )
    output = value.new_empty(-(-len(value) // 256))
    load("reduce").complex_energy(
        value=value, reference=reference, output=output, count=len(value)
    ).launchRaw(**blocks(len(value)))
    return output.sum()

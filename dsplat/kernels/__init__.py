"""Slang kernels and their PyTorch bindings; the only code that launches kernels.

Each .slang file has one binding module: spectral, surfel (surfel.slang and
correlate.slang), search, reduce and response; cublas binds the symmetric rank-k
update that PyTorch lacks. Tensors are CUDA float32, complex
values complex64, and complex gradients follow the PyTorch convention
dL/dx = Re(conj(upstream) * dy/dx).
"""

from functools import cache
from pathlib import Path

import slangtorch
import torch


@cache
def load(name: str, **defines: int):
    """Compile or reuse one kernel module; defines select a compile-time specialization."""
    path = Path(__file__).parent / f"{name}.slang"
    return slangtorch.loadModule(str(path), defines=defines)


def blocks(count: int, size: int = 256) -> dict:
    """launchRaw arguments for one thread per item in a 1D grid."""
    return dict(blockSize=(size, 1, 1), gridSize=(-(-count // size), 1, 1))


def interleaved(value):
    """Interleaved real/imaginary view [..., 2] of a complex tensor, as the kernels read it."""
    return torch.view_as_real(value.to(torch.complex64).resolve_conj().resolve_neg()).contiguous()

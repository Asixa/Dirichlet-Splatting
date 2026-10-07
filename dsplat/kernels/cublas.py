"""cuBLAS symmetric rank-k update, which PyTorch does not expose on CUDA."""

import ctypes
from functools import cache
from pathlib import Path

import torch


@cache
def _library(device: int):
    """The cuBLAS library shipped with PyTorch and a handle on this device.

    Windows wheels ship it in torch/lib, Linux wheels in nvidia/cublas.
    """
    root = Path(torch.__file__).parent
    patterns = ("lib/cublas64_*.dll", "lib/libcublas.so*", "../nvidia/cublas/lib/libcublas.so*")
    name = next(path for pattern in patterns for path in sorted(root.glob(pattern)))
    library, handle = ctypes.CDLL(str(name)), ctypes.c_void_p()
    with torch.cuda.device(device):
        library.cublasCreate_v2(ctypes.byref(handle))
    real, integer, pointer = ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
    library.cublasSetStream_v2.argtypes = [pointer, pointer]
    library.cublasSsyrk_v2.argtypes = [
        pointer, integer, integer, integer, integer, real, pointer, integer, real, pointer,
        integer,
    ]  # fmt: skip
    return library, handle


def syrk(rows, out):
    """Add rows^T rows to the upper triangle of out; the lower triangle is untouched.

    cuBLAS sees the row-major [m, n] block as a column-major [n, m] matrix, and its
    lower triangle of the result is the upper triangle of the torch tensor.
    """
    library, handle = _library(rows.device.index or 0)
    library.cublasSetStream_v2(handle, torch.cuda.current_stream(rows.device).cuda_stream)
    count, columns = rows.shape
    one = ctypes.c_float(1.0)
    library.cublasSsyrk_v2(
        handle, 0, 0, columns, count, ctypes.byref(one), rows.data_ptr(), columns,
        ctypes.byref(one), out.data_ptr(), columns,
    )  # fmt: skip

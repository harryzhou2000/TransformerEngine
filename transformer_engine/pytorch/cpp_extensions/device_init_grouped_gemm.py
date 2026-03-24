# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Python interface for device-initiated CUTLASS grouped GEMM.

This is a standalone wrapper for the MXFP8 device-initiated grouped GEMM kernel,
separated from the main cuBLAS-based grouped GEMM path to avoid interfering with
existing functionality on main.
"""

from typing import List, Optional, Tuple, Union
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor

from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

__all__ = [
    "device_init_grouped_gemm",
    "get_device_init_grouped_gemm_workspace",
]

# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

_cutlass_device_init_workspace: Optional[List[torch.Tensor]] = None


def get_cutlass_device_init_workspace_size_bytes() -> int:
    """16 MiB device workspace for CUTLASS device-initiated grouped GEMM."""
    return 16_777_216


def get_device_init_grouped_gemm_workspace() -> List[torch.Tensor]:
    """Returns pre-allocated workspace for device-initiated CUTLASS grouped GEMM.

    Returns a list of two tensors:
      [0] Device buffer for CUTLASS arguments and kernel workspace (16 MiB)
      [1] Host pinned buffer for async H2D copy of weight pointers (graph-safe)
    """
    global _cutlass_device_init_workspace
    if _cutlass_device_init_workspace is None:
        _cutlass_device_init_workspace = [
            # Device buffer for cutlass arguments and kernel
            torch.empty(
                get_cutlass_device_init_workspace_size_bytes(),
                dtype=torch.uint8,
                device="cuda",
            ),
            # Host pinned buffer for the source of H2D copy of cutlass arguments.
            # CUDA Graph capture does not support .pinned_memory(), so a global workspace
            # is needed.
            torch.empty(
                int(os.getenv("NVTE_CUTLASS_HOST_PINNED_U64_CAPACITY", "4194304")),
                dtype=torch.uint64,
                device="cpu",
                pin_memory=True,
            ),
        ]
    return _cutlass_device_init_workspace


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def device_init_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[torch.Tensor] = None,
    gelu: bool = False,
    grad: bool = False,
    wgrad: bool = False,
    accumulate: bool = False,
    accumulate_mask: Optional[torch.Tensor] = None,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output: bool = False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    Device-initiated CUTLASS grouped GEMM for MoE with MXFP8 inputs.

    m_splits must be a CUDA int64 tensor with per-expert token counts.
    A and B must be MXFP8TensorStorage.
    """
    if isinstance(m_splits, list):
        m_splits = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
    assert m_splits is not None and m_splits.is_cuda, "m_splits must be a CUDA tensor"

    num_gemms = m_splits.size(0)
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    assert isinstance(A[0], MXFP8TensorStorage) and isinstance(
        B[0], MXFP8TensorStorage
    ), "Only MXFP8 A and B are supported for device-initiated grouped GEMM"
    assert (
        out[0].dtype == torch.bfloat16
        or out[0].dtype == torch.float16
        or (wgrad and out[0].dtype == torch.float32)
    ), "Only BF16, FP16, or FP32 (wgrad only) output is supported"
    assert not use_bias, "Bias is not supported for device-initiated grouped GEMM"
    assert not gelu, "GELU is not supported for device-initiated grouped GEMM"

    gelu_input = empty_tensors
    out_dtype_te = TE_DType[out[0].dtype] if D_dtype is None else D_dtype
    assert TE_DType[out[0].dtype] == out_dtype_te, (
        f"Output dtype mismatch: out[0].dtype={out[0].dtype}, out_dtype={out_dtype_te}"
    )

    bias_list = bias if use_bias else empty_tensors
    bias_dtype = TE_DType[torch.bfloat16]

    sm_count = get_sm_count()

    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda")
            for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors

    result_bias = tex.te_general_device_init_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype_te,
        m_splits,
        grad_bias if grad else bias_list,
        bias_dtype,
        single_output,
        gelu_input,
        grad,
        wgrad,
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        accumulate_mask,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, result_bias, gelu_input

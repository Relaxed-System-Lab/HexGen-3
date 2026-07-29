from contextlib import contextmanager
from typing import Optional

try:
    import torch
    from torch.cuda import nvtx as torch_nvtx
except Exception:  # pragma: no cover - fallback when torch/cuda not available
    torch = None
    torch_nvtx = None


@contextmanager
def nvtx_range(name: str):
    """Lightweight NVTX range helper; no-op if CUDA/NVTX unavailable."""
    if torch is not None and torch_nvtx is not None and torch.cuda.is_available():
        torch_nvtx.range_push(name)
        try:
            yield
        finally:
            torch_nvtx.range_pop()
    else:
        yield


def format_batch_size_info(
    batch_size: Optional[int] = None,
    num_tokens: Optional[int] = None,
    hidden_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> str:
    """
    Format batch size information for NVTX range names.
    
    Args:
        batch_size: Number of requests in the batch
        num_tokens: Total number of tokens (seq_lens_sum)
        hidden_size: Hidden dimension size
        max_seq_len: Maximum sequence length in the batch
    
    Returns:
        Formatted string like "[b=2,t=128,h=2048]" or empty string if no info provided
    """
    parts = []
    if batch_size is not None:
        parts.append(f"b={batch_size}")
    if num_tokens is not None:
        parts.append(f"t={num_tokens}")
    if hidden_size is not None:
        parts.append(f"h={hidden_size}")
    if max_seq_len is not None:
        parts.append(f"max_s={max_seq_len}")
    
    if parts:
        return "[" + ",".join(parts) + "]"
    return ""


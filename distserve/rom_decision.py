"""
Cost-estimation and decision logic for the RoM scheduler.

Formulas
--------
KV cache size (bytes)
    kv_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
    (factor-2: one tensor for K, one for V)

Migration cost (seconds)
    C_mig = kv_bytes / (bandwidth_mbps × 10⁶)

Recomputation cost (seconds)
    C_recomp = linear_interpolation(prefill_time_table, prompt_len)

Decision
    policy "rom"               → argmin(C_mig, C_recomp)
    policy "always_migrate"    → "migrate"
    policy "always_recompute"  → "recompute"
"""
from __future__ import annotations

import bisect
import math
from typing import Dict, Literal, Tuple

from distserve.rom_config import ROMConfig

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------
DecisionStr = Literal["migrate", "recompute"]


# ===========================================================================
# KV cache sizing
# ===========================================================================

def compute_kv_size_bytes(
    prompt_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Compute the uncompressed KV-cache size in bytes for a single request.

    This matches the layout used by ParaWorker.init_kvcache_and_swap():
        [num_blocks, num_layers, num_heads, block_size, head_dim]
    where block_size × block_count = seq_len.  The per-token footprint
    simplifies to the formula below (block granularity cancels out).

    Parameters
    ----------
    prompt_len   : number of tokens in the prompt (= sequence length)
    num_layers   : number of transformer layers
    num_kv_heads : number of K/V attention heads per layer
    head_dim     : per-head feature dimension
    dtype_bytes  : bytes per element (2 for fp16/bf16, 4 for fp32)

    Returns
    -------
    int
        Byte count, always ≥ 0.
    """
    if prompt_len <= 0:
        return 0
    return 2 * num_layers * num_kv_heads * head_dim * prompt_len * dtype_bytes


def compute_kv_size_bytes_from_config(
    prompt_len: int,
    rom_config: ROMConfig,
) -> int:
    """
    Reads architecture parameters from *rom_config*.
    """
    return compute_kv_size_bytes(
        prompt_len,
        num_layers=rom_config.num_layers,
        num_kv_heads=rom_config.num_kv_heads,
        head_dim=rom_config.head_dim,
        dtype_bytes=rom_config.dtype_bytes,
    )


# ===========================================================================
# Migration cost
# ===========================================================================

def compute_c_mig(kv_size_bytes: int, bandwidth_mbps: float) -> float:
    """
    Estimate the time (seconds) to transfer *kv_size_bytes* over a link whose
    throughput is *bandwidth_mbps* MB/s.

    Returns +∞ if bandwidth is zero or negative (link is down).
    """
    if bandwidth_mbps <= 0 or kv_size_bytes < 0:
        return math.inf
    if kv_size_bytes == 0:
        return 0.0
    return kv_size_bytes / (bandwidth_mbps * 1_000_000.0)


# ===========================================================================
# Recomputation cost  (prefill-time lookup)
# ===========================================================================

def _interpolate(keys: list, values: list, x: int) -> float:
    """
    Linear interpolation / flat extrapolation over a sorted table.

    Parameters
    ----------
    keys   : sorted list of prompt-length breakpoints
    values : corresponding measured latencies
    x      : query prompt length
    """
    if x <= keys[0]:
        return float(values[0])
    if x >= keys[-1]:
        return float(values[-1])
    idx = bisect.bisect_right(keys, x)
    lo, hi = keys[idx - 1], keys[idx]
    t_lo, t_hi = values[idx - 1], values[idx]
    frac = (x - lo) / (hi - lo)
    return t_lo + frac * (t_hi - t_lo)


def compute_c_recomp(
    prompt_len: int,
    prefill_time_table: Dict[int, float],
) -> float:
    """
    Estimate the recomputation cost (seconds) as the predicted prefill latency
    for a prompt of *prompt_len* tokens, using linear interpolation over the
    pre-profiled *prefill_time_table*.

    Parameters
    ----------
    prompt_len         : number of tokens in the prompt
    prefill_time_table : dict mapping prompt-length buckets to latency (s).
                         Must have at least one entry.

    Raises
    ------
    ValueError if *prefill_time_table* is empty.
    """
    if not prefill_time_table:
        raise ValueError("prefill_time_table must not be empty")
    if prompt_len <= 0:
        return 0.0

    keys = sorted(prefill_time_table.keys())
    values = [prefill_time_table[k] for k in keys]
    return _interpolate(keys, values, prompt_len)


# ===========================================================================
# Decision function
# ===========================================================================

def make_decision(
    prompt_len: int,
    bandwidth_mbps: float,
    rom_config: ROMConfig,
) -> Tuple[DecisionStr, float, float]:
    """
    Decide whether to migrate the existing KV cache or to recompute it.

    Parameters
    ----------
    prompt_len     : number of tokens in the original prompt
    bandwidth_mbps : current inter-node bandwidth estimate (MB/s);
                     typically provided by LocalScheduler.get_bandwidth()
    rom_config     : fully initialised ROMConfig

    Returns
    -------
    (decision, C_mig, C_recomp)
        decision  : "migrate" or "recompute"
        C_mig     : estimated migration latency (seconds)
        C_recomp  : estimated recomputation latency (seconds)

    Notes
    -----
    * If *bandwidth_mbps* is 0 or negative,
      C_mig is +∞ and the function always chooses "recompute" under the
      "rom" policy.
    * Under "always_migrate" / "always_recompute" policies the costs are still
      computed and returned for logging, but the decision is forced.
    """
    kv_bytes = compute_kv_size_bytes_from_config(prompt_len, rom_config)
    c_mig = compute_c_mig(kv_bytes, bandwidth_mbps)
    c_recomp = compute_c_recomp(prompt_len, rom_config.prefill_time_table)

    if rom_config.policy == "always_migrate":
        decision: DecisionStr = "migrate"
    elif rom_config.policy == "always_recompute":
        decision = "recompute"
    else:
        # "rom" policy: choose the lower-cost path; prefer migrate on a tie
        decision = "migrate" if c_mig <= c_recomp else "recompute"

    return decision, c_mig, c_recomp


# ===========================================================================
# Breakeven analysis helper
# ===========================================================================

def breakeven_prompt_len(
    bandwidth_mbps: float,
    rom_config: ROMConfig,
) -> int:
    """
    Find the smallest prompt length at which recomputing becomes cheaper than
    migrating. Uses linear search over the table buckets; exact only at table breakpoints.

    Returns
    -------
    int
        Crossover prompt length in tokens, or 0 if migration is always cheaper,
        or sys.maxsize if recompute is always cheaper.
    """
    import sys
    keys = sorted(rom_config.prefill_time_table.keys())
    for k in keys:
        _, c_mig, c_recomp = make_decision(k, bandwidth_mbps, rom_config)
        if c_recomp <= c_mig:
            return k
    return sys.maxsize
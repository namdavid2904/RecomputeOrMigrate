"""
A Ray actor that aggregates inter-node bandwidth gossip and
per-worker decode-queue depths, then provides the best target worker for
RoM recovery placement.

Design notes
------------
* One LocalScheduler actor is started per DistServe deployment (not per
  node) and is accessible from any Ray task in the cluster.
* Bandwidth is updated via push: whoever measures the link (a periodic
  probe coroutine in LLMEngine) calls update_bandwidth().
* Queue depths are updated via push: DecodingStageLLMEngine calls
  update_queue_length() after each scheduling step.
* The actor is single-threaded (Ray default), so no locking is needed.
* When no measurement has been received recently, the actor falls back to
  rom_config.sim_bandwidth_mbps so the decision module always gets a
  valid number.

Usage (inside LLMEngine)
------------------------
    from distserve.rom_local_scheduler import LocalScheduler

    sched = LocalScheduler.options(name="rom_local_scheduler").remote(rom_cfg)

    # Register workers at startup
    for wid in range(num_decode_workers):
        ray.get(sched.register_worker.remote(wid))

    # Update bandwidth from a probing task
    sched.update_bandwidth.remote(measured_mbps)

    # Update queue depth after each decode step
    sched.update_queue_length.remote(worker_id=0, length=engine.queue_size())

    # ROM decision path
    state  = ray.get(sched.get_state.remote())
    bw     = state["bandwidth_mbps"]
    target = ray.get(sched.get_best_decode_worker.remote(exclude=[failed_wid]))
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import ray

from distserve.rom_config import ROMConfig

# How long (seconds) a bandwidth measurement remains "fresh".
# Beyond this threshold the fallback value is returned.
_BW_STALENESS_THRESHOLD_S = 60.0


@ray.remote
class LocalScheduler:
    """
    Ray actor: gossip bandwidth tracker + decode-queue load balancer.

    Parameters
    ----------
    rom_config : ROMConfig
        Used to seed the bandwidth EWMA and to provide the fallback value.
    """

    def __init__(self, rom_config: ROMConfig) -> None:
        self._config = rom_config

        # ── Bandwidth tracking ─────────────────────────────────────────────
        # Seeded with the configured fallback so decisions are valid
        # even before the first measurement arrives.
        self._bandwidth_mbps: float = rom_config.sim_bandwidth_mbps
        # EWMA smoothing coefficient (0 < α ≤ 1; smaller = slower adaptation)
        self._bw_alpha: float = 0.2
        self._last_bw_update: float = -math.inf   # "never updated"

        # ── Queue-length tracking  ─────────────────────────────────────────
        # worker_id → pending request count
        self._queue_lengths: Dict[int, int] = {}

        # ── History (ring buffer) for diagnostics  ─────────────────────────
        self._bw_history: List[float] = []
        self._bw_history_max = 100

    # ======================================================================
    # Bandwidth gossip API
    # ======================================================================

    def update_bandwidth(self, bandwidth_mbps: float) -> None:
        """
        Ingest a new bandwidth measurement (MB/s) and update the EWMA.

        Silently ignores non-positive values (probe error / link down).
        """
        if bandwidth_mbps <= 0:
            return
        α = self._bw_alpha
        self._bandwidth_mbps = α * bandwidth_mbps + (1 - α) * self._bandwidth_mbps
        self._last_bw_update = time.monotonic()

        # Keep a rolling history for offline analysis
        self._bw_history.append(bandwidth_mbps)
        if len(self._bw_history) > self._bw_history_max:
            self._bw_history.pop(0)

    def get_bandwidth(self) -> float:
        """
        Return the current bandwidth estimate (MB/s).

        Returns the EWMA value if a measurement was received within the last
        60 seconds; otherwise falls back to rom_config.sim_bandwidth_mbps.
        """
        age = time.monotonic() - self._last_bw_update
        if age > _BW_STALENESS_THRESHOLD_S:
            return self._config.sim_bandwidth_mbps
        return self._bandwidth_mbps

    def set_bandwidth_alpha(self, alpha: float) -> None:
        """Adjust the EWMA smoothing coefficient at runtime (testing / tuning)."""
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._bw_alpha = alpha

    # ======================================================================
    # Decode-worker queue management API
    # ======================================================================

    def register_worker(self, worker_id: int) -> None:
        """
        Register a worker so it appears in the candidate pool immediately,
        even before any queue-length update has arrived.
        """
        self._queue_lengths.setdefault(worker_id, 0)

    def deregister_worker(self, worker_id: int) -> None:
        """
        Remove a failed or decommissioned worker from the candidate pool so
        it is never returned as a target.
        """
        self._queue_lengths.pop(worker_id, None)

    def update_queue_length(self, worker_id: int, length: int) -> None:
        """
        Record the current number of queued requests at *worker_id*.

        Negative values are clamped to 0.
        """
        self._queue_lengths[worker_id] = max(0, length)

    def get_queue_length(self, worker_id: int) -> int:
        """Return the last-known queue depth for *worker_id* (0 if unknown)."""
        return self._queue_lengths.get(worker_id, 0)

    def get_best_decode_worker(
        self,
        exclude: Optional[List[int]] = None,
    ) -> Optional[int]:
        """
        Return the *worker_id* with the shortest pending queue, excluding any
        ids in *exclude* (typically the failed worker).

        Returns None if no eligible workers are registered.
        """
        exclude_set = set(exclude or [])
        candidates = {
            wid: qlen
            for wid, qlen in self._queue_lengths.items()
            if wid not in exclude_set
        }
        if not candidates:
            return None
        return min(candidates, key=lambda w: candidates[w])

    # ======================================================================
    # State snapshot  (for decision helper and monitoring)
    # ======================================================================

    def get_state(self) -> dict:
        """
        Return a serialisable dict snapshot of the current scheduler state.

        Keys
        ----
        bandwidth_mbps        : current bandwidth estimate (MB/s)
        bw_fresh              : True if estimate is based on a recent measurement
        queue_lengths         : {worker_id: queue_depth}
        bw_history            : recent raw bandwidth samples (ring buffer)
        last_bw_update_age_s  : seconds since the last bandwidth update
        """
        age = time.monotonic() - self._last_bw_update
        return {
            "bandwidth_mbps": self.get_bandwidth(),
            "bw_fresh": age <= _BW_STALENESS_THRESHOLD_S,
            "queue_lengths": dict(self._queue_lengths),
            "bw_history": list(self._bw_history),
            "last_bw_update_age_s": age if age != math.inf else None,
        }

    # ======================================================================
    # Testing / admin helpers
    # ======================================================================

    def reset(self) -> None:
        """
        Reset all measurements to initial defaults.
        Used in tests to get a clean slate without restarting the actor.
        """
        self._bandwidth_mbps = self._config.sim_bandwidth_mbps
        self._last_bw_update = -math.inf
        self._queue_lengths.clear()
        self._bw_history.clear()
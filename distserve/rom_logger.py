"""
JSONL logger for all RoM scheduler events.

Three event types are emitted:

``rom_decision``
    One record per request per failure event, capturing the prompt length,
    bandwidth estimate, both cost estimates, and the final decision.

``rom_recovery``
    One record per recovery attempt, capturing whether it succeeded and
    how long it took.

``decode_worker_failure``
    One record when the health monitor first detects a failed worker.

Usage
-----
    from distserve.rom_logger import ROMLogger

    rom_log = ROMLogger(log_file="/tmp/rom_decisions.jsonl")

    rom_log.log_decision(
        request_id="req-42",
        prompt_len=512,
        bandwidth_mbps=9800.0,
        c_mig=0.003,
        c_recomp=0.032,
        decision="migrate",
        policy="rom",
        failed_worker_id=2,
        target_worker_id=1,
    )

    rom_log.log_recovery(
        request_id="req-42",
        decision="migrate",
        success=True,
        latency_s=0.004,
    )
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

# ── Logger name used by the Python logging hierarchy ────────────────────────
ROM_LOGGER_NAME = "distserve.rom"


# ===========================================================================
# Low-level Python logger setup
# ===========================================================================

def _make_file_handler(path: str) -> logging.FileHandler:
    """Create a FileHandler that appends raw JSON lines to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def get_python_logger(log_file: Optional[str] = None) -> logging.Logger:
    """
    Return the singleton ROM Python logger, configuring it on first call.

    Parameters
    ----------
    log_file : str or None
        If provided and the logger has no file handler yet, a JSONL file
        handler is attached.  Subsequent calls with the same or different
        path do NOT re-attach (call reset_python_logger() first if needed).
    """
    logger = logging.getLogger(ROM_LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(logging.Formatter("[ROM %(levelname)s] %(message)s"))
        logger.addHandler(sh)

    # Attach file handler if requested and not already there
    if log_file and not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        logger.addHandler(_make_file_handler(log_file))

    return logger


def reset_python_logger() -> None:
    """
    Remove all handlers from the ROM logger.
    Useful in tests to avoid handler accumulation across test cases.
    """
    logger = logging.getLogger(ROM_LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


# ===========================================================================
# ROMLogger class
# ===========================================================================

class ROMLogger:
    """
    Serialises RoM scheduler events to JSONL.

    All ``log_*`` methods are synchronous, cheap, and safe to call from
    inside an asyncio event loop (they do not block on I/O for long).

    Parameters
    ----------
    log_file : str or None
        Path for the JSONL output file.  Pass None to log only to stderr
        (WARNING+ level).
    """

    def __init__(self, log_file: Optional[str] = None) -> None:
        self._logger = get_python_logger(log_file)
        self._log_file = log_file

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def log_decision(
        self,
        *,
        request_id: str,
        prompt_len: int,
        bandwidth_mbps: float,
        c_mig: float,
        c_recomp: float,
        decision: str,
        policy: str,
        failed_worker_id: Optional[int] = None,
        target_worker_id: Optional[int] = None,
    ) -> None:
        """
        Emit one ``rom_decision`` record.

        Parameters
        ----------
        request_id       : unique request identifier (from LLMEngine)
        prompt_len       : number of tokens in the prompt
        bandwidth_mbps   : bandwidth estimate used for the decision (MB/s)
        c_mig            : estimated migration cost (seconds)
        c_recomp         : estimated recomputation cost (seconds)
        decision         : "migrate" or "recompute"
        policy           : active ROM policy string
        failed_worker_id : index of the failed decode worker
        target_worker_id : chosen recovery target worker index
        """
        record = {
            "event": "rom_decision",
            "ts": time.time(),
            "request_id": request_id,
            "prompt_len": prompt_len,
            "bandwidth_mbps": round(bandwidth_mbps, 3),
            "c_mig_s": round(c_mig, 6),
            "c_recomp_s": round(c_recomp, 6),
            "decision": decision,
            "policy": policy,
            "failed_worker_id": failed_worker_id,
            "target_worker_id": target_worker_id,
        }
        self._logger.info(json.dumps(record))

    def log_recovery(
        self,
        *,
        request_id: str,
        decision: str,
        success: bool,
        latency_s: float,
        error: Optional[str] = None,
    ) -> None:
        """
        Emit one ``rom_recovery`` record after a recovery action completes.

        Parameters
        ----------
        request_id : unique request identifier
        decision   : "migrate" or "recompute" (matches the prior decision record)
        success    : True if recovery completed without error
        latency_s  : wall-clock time from decision to recovery completion
        error      : exception message if success is False
        """
        record = {
            "event": "rom_recovery",
            "ts": time.time(),
            "request_id": request_id,
            "decision": decision,
            "success": success,
            "latency_s": round(latency_s, 6),
            "error": error,
        }
        level = logging.INFO if success else logging.WARNING
        self._logger.log(level, json.dumps(record))

    def log_worker_failure(
        self,
        *,
        worker_id: int,
        affected_request_ids: List[str],
        error_msg: str,
    ) -> None:
        """
        Emit one ``decode_worker_failure`` record on first detection.

        Parameters
        ----------
        worker_id             : index of the failed worker
        affected_request_ids  : request ids that were in-flight on that worker
        error_msg             : stringified RayActorError
        """
        record = {
            "event": "decode_worker_failure",
            "ts": time.time(),
            "worker_id": worker_id,
            "affected_request_count": len(affected_request_ids),
            "affected_request_ids": affected_request_ids,
            "error": error_msg,
        }
        self._logger.warning(json.dumps(record))

    # ------------------------------------------------------------------ #
    # Testing helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def log_file(self) -> Optional[str]:
        """Return the configured log file path."""
        return self._log_file
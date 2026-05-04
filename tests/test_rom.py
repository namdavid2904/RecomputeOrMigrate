"""
Unit tests for the RoM scheduler modules.

Coverage
--------
1. rom_config   — field defaults, env-var overrides, validation
2. rom_decision — KV size formula, cost functions, interpolation,
                  decision policies, breakeven helper
3. rom_logger   — record emission, JSONL format, file I/O
4. rom_local_scheduler — EWMA, queue management, staleness fallback
                         (Ray-free: uses unittest.mock to patch ray.remote)

Run with::
    python -m pytest tests/test_rom.py -v
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distserve.rom_config import ROMConfig, VALID_POLICIES
from distserve.rom_decision import (
    breakeven_prompt_len,
    compute_c_mig,
    compute_c_recomp,
    compute_kv_size_bytes,
    compute_kv_size_bytes_from_config,
    make_decision,
)
from distserve.rom_logger import ROMLogger, reset_python_logger


# ===========================================================================
# ROMConfig tests
# ===========================================================================

class TestROMConfig(unittest.TestCase):

    def tearDown(self):
        # Clean env-var leakage between tests
        for key in list(os.environ.keys()):
            if key.startswith("ROM_"):
                del os.environ[key]

    # ── Defaults ─────────────────────────────────────────────────────────────

    def test_default_policy(self):
        cfg = ROMConfig()
        self.assertEqual(cfg.policy, "rom")

    def test_default_bandwidth(self):
        cfg = ROMConfig()
        self.assertEqual(cfg.sim_bandwidth_mbps, 10_000.0)

    def test_default_monitor_enabled(self):
        cfg = ROMConfig()
        self.assertTrue(cfg.monitor_enabled)

    def test_default_prefill_table_not_empty(self):
        cfg = ROMConfig()
        self.assertGreater(len(cfg.prefill_time_table), 0)

    # ── from_env ─────────────────────────────────────────────────────────────

    def test_from_env_policy_override(self):
        os.environ["ROM_POLICY"] = "always_migrate"
        cfg = ROMConfig.from_env()
        self.assertEqual(cfg.policy, "always_migrate")

    def test_from_env_bandwidth_override(self):
        os.environ["ROM_SIM_BANDWIDTH_MBPS"] = "500.0"
        cfg = ROMConfig.from_env()
        self.assertAlmostEqual(cfg.sim_bandwidth_mbps, 500.0)

    def test_from_env_monitor_disabled(self):
        os.environ["ROM_MONITOR_ENABLED"] = "0"
        cfg = ROMConfig.from_env()
        self.assertFalse(cfg.monitor_enabled)

    def test_from_env_monitor_enabled_by_string(self):
        os.environ["ROM_MONITOR_ENABLED"] = "false"
        cfg = ROMConfig.from_env()
        self.assertFalse(cfg.monitor_enabled)

    def test_from_env_num_layers(self):
        os.environ["ROM_NUM_LAYERS"] = "40"
        cfg = ROMConfig.from_env()
        self.assertEqual(cfg.num_layers, 40)

    def test_from_env_invalid_policy_raises(self):
        os.environ["ROM_POLICY"] = "banana"
        with self.assertRaises(ValueError):
            ROMConfig.from_env()

    # ── validate ─────────────────────────────────────────────────────────────

    def test_validate_bad_bandwidth(self):
        cfg = ROMConfig(sim_bandwidth_mbps=0.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_bad_dtype_bytes(self):
        cfg = ROMConfig(dtype_bytes=3)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_empty_prefill_table(self):
        cfg = ROMConfig(prefill_time_table={})
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_all_valid_policies(self):
        for policy in VALID_POLICIES:
            cfg = ROMConfig(policy=policy)
            cfg.validate()  # should not raise

    # ── update_from_model_config ──────────────────────────────────────────────

    def test_update_from_model_config(self):
        cfg = ROMConfig()
        cfg.update_from_model_config(num_layers=40, num_kv_heads=40, head_dim=128)
        self.assertEqual(cfg.num_layers, 40)
        self.assertEqual(cfg.num_kv_heads, 40)


# ===========================================================================
# compute_kv_size_bytes tests
# ===========================================================================

class TestComputeKVSizeBytes(unittest.TestCase):

    def test_zero_prompt_len(self):
        self.assertEqual(compute_kv_size_bytes(0, 32, 32, 128, 2), 0)

    def test_negative_prompt_len(self):
        self.assertEqual(compute_kv_size_bytes(-5, 32, 32, 128, 2), 0)

    def test_formula_correctness(self):
        # 2 * 32 layers * 32 heads * 128 dim * 512 tokens * 2 bytes = 536 870 912
        expected = 2 * 32 * 32 * 128 * 512 * 2
        result = compute_kv_size_bytes(512, 32, 32, 128, 2)
        self.assertEqual(result, expected)

    def test_fp32(self):
        # dtype_bytes=4 should double compared to fp16
        fp16 = compute_kv_size_bytes(256, 32, 32, 128, 2)
        fp32 = compute_kv_size_bytes(256, 32, 32, 128, 4)
        self.assertEqual(fp32, 2 * fp16)

    def test_from_config(self):
        cfg = ROMConfig(num_layers=12, num_kv_heads=12, head_dim=64, dtype_bytes=2)
        expected = compute_kv_size_bytes(256, 12, 12, 64, 2)
        self.assertEqual(compute_kv_size_bytes_from_config(256, cfg), expected)

    def test_scales_linearly_with_prompt_len(self):
        a = compute_kv_size_bytes(128, 32, 32, 128, 2)
        b = compute_kv_size_bytes(256, 32, 32, 128, 2)
        self.assertEqual(b, 2 * a)


# ===========================================================================
# compute_c_mig tests
# ===========================================================================

class TestComputeCMig(unittest.TestCase):

    def test_zero_bandwidth_returns_inf(self):
        self.assertEqual(compute_c_mig(1_000_000, 0.0), math.inf)

    def test_negative_bandwidth_returns_inf(self):
        self.assertEqual(compute_c_mig(1_000_000, -1.0), math.inf)

    def test_zero_kv_size(self):
        self.assertEqual(compute_c_mig(0, 10_000.0), 0.0)

    def test_formula(self):
        # 1 GB at 1 GB/s = 1 second
        result = compute_c_mig(1_000_000_000, 1_000.0)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_high_bandwidth_is_cheap(self):
        # 10 GB/s: 100 MB cache should take ~10 ms
        result = compute_c_mig(100 * 1024 * 1024, 10_000.0)
        self.assertLess(result, 0.015)

    def test_low_bandwidth_is_expensive(self):
        # 10 MB/s: 100 MB cache takes ~10 s
        result = compute_c_mig(100 * 1024 * 1024, 10.0)
        self.assertGreater(result, 5.0)


# ===========================================================================
# compute_c_recomp / interpolation tests
# ===========================================================================

class TestComputeCRecomp(unittest.TestCase):

    TABLE = {128: 0.010, 512: 0.032, 1024: 0.060}

    def test_exact_key(self):
        self.assertAlmostEqual(
            compute_c_recomp(512, self.TABLE), 0.032, places=6
        )

    def test_below_table_minimum(self):
        # Should clamp to first value
        self.assertAlmostEqual(
            compute_c_recomp(1, self.TABLE), 0.010, places=6
        )

    def test_above_table_maximum(self):
        # Should clamp to last value
        self.assertAlmostEqual(
            compute_c_recomp(4096, self.TABLE), 0.060, places=6
        )

    def test_interpolation_midpoint(self):
        # At 320 tokens (midpoint of 128–512 range)
        # frac = (320-128)/(512-128) = 192/384 = 0.5
        expected = 0.010 + 0.5 * (0.032 - 0.010)
        result = compute_c_recomp(320, self.TABLE)
        self.assertAlmostEqual(result, expected, places=6)

    def test_interpolation_quarter(self):
        # At 224 tokens (25% between 128 and 512)
        frac = (224 - 128) / (512 - 128)
        expected = 0.010 + frac * (0.032 - 0.010)
        result = compute_c_recomp(224, self.TABLE)
        self.assertAlmostEqual(result, expected, places=6)

    def test_zero_prompt_len(self):
        self.assertEqual(compute_c_recomp(0, self.TABLE), 0.0)

    def test_empty_table_raises(self):
        with self.assertRaises(ValueError):
            compute_c_recomp(512, {})


# ===========================================================================
# make_decision tests
# ===========================================================================

class TestMakeDecision(unittest.TestCase):

    def _cfg(self, policy="rom", bw=10_000.0):
        return ROMConfig(
            policy=policy,
            sim_bandwidth_mbps=bw,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            dtype_bytes=2,
            prefill_time_table={128: 0.010, 512: 0.032, 1024: 0.060},
        )

    def test_policy_always_migrate(self):
        cfg = self._cfg(policy="always_migrate")
        decision, c_mig, c_recomp = make_decision(512, 10.0, cfg)
        self.assertEqual(decision, "migrate")

    def test_policy_always_recompute(self):
        cfg = self._cfg(policy="always_recompute")
        decision, c_mig, c_recomp = make_decision(512, 10_000.0, cfg)
        self.assertEqual(decision, "recompute")

    def test_rom_chooses_migrate_when_bw_high(self):
        cfg = self._cfg(policy="rom", bw=10_000.0)
        # At 10 GB/s: small cache should be cheap to migrate
        decision, c_mig, c_recomp = make_decision(64, 10_000.0, cfg)
        self.assertEqual(decision, "migrate")
        self.assertLessEqual(c_mig, c_recomp)

    def test_rom_chooses_recompute_when_bw_low(self):
        cfg = self._cfg(policy="rom", bw=10.0)
        # At 10 MB/s: large cache is very slow to transfer
        decision, c_mig, c_recomp = make_decision(1024, 10.0, cfg)
        self.assertEqual(decision, "recompute")
        self.assertLess(c_recomp, c_mig)

    def test_zero_bandwidth_always_recomputes(self):
        cfg = self._cfg(policy="rom")
        decision, c_mig, c_recomp = make_decision(512, 0.0, cfg)
        self.assertEqual(decision, "recompute")
        self.assertEqual(c_mig, math.inf)

    def test_costs_always_returned(self):
        cfg = self._cfg(policy="always_migrate")
        decision, c_mig, c_recomp = make_decision(512, 10.0, cfg)
        # Costs should be computed regardless of policy
        self.assertGreater(c_mig, 0)
        self.assertGreater(c_recomp, 0)

    def test_tie_prefers_migrate(self):
        """When C_mig == C_recomp, migrate should be chosen."""
        cfg = self._cfg(policy="rom")
        # Patch the decision function to force a tie
        with patch(
            "distserve.rom_decision.compute_c_mig", return_value=0.032
        ), patch(
            "distserve.rom_decision.compute_c_recomp", return_value=0.032
        ):
            decision, c_mig, c_recomp = make_decision(512, 10_000.0, cfg)
        self.assertEqual(decision, "migrate")


# ===========================================================================
# breakeven_prompt_len tests
# ===========================================================================

class TestBreakevenPromptLen(unittest.TestCase):

    def test_high_bandwidth_no_breakeven(self):
        """At very high bandwidth, recompute is never cheaper within the table."""
        cfg = ROMConfig(
            policy="rom",
            sim_bandwidth_mbps=100_000.0,
            num_layers=32, num_kv_heads=32, head_dim=128, dtype_bytes=2,
            prefill_time_table={128: 0.010, 512: 0.032, 1024: 0.060, 4096: 0.225},
        )
        result = breakeven_prompt_len(100_000.0, cfg)
        import sys
        self.assertEqual(result, sys.maxsize)

    def test_low_bandwidth_breakeven_at_small_len(self):
        """At very low bandwidth, recompute wins even for short prompts."""
        cfg = ROMConfig(
            policy="rom",
            sim_bandwidth_mbps=1.0,
            num_layers=32, num_kv_heads=32, head_dim=128, dtype_bytes=2,
            prefill_time_table={128: 0.010, 512: 0.032, 1024: 0.060},
        )
        result = breakeven_prompt_len(1.0, cfg)
        # Should be ≤ 128 (migrate is already expensive at 1 MB/s)
        self.assertLessEqual(result, 128)


# ===========================================================================
# ROMLogger tests
# ===========================================================================

class TestROMLogger(unittest.TestCase):

    def setUp(self):
        reset_python_logger()

    def tearDown(self):
        reset_python_logger()

    def test_log_decision_writes_jsonl(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            logger = ROMLogger(log_file=log_path)
            logger.log_decision(
                request_id="req-1",
                prompt_len=512,
                bandwidth_mbps=9500.0,
                c_mig=0.003,
                c_recomp=0.032,
                decision="migrate",
                policy="rom",
                failed_worker_id=2,
                target_worker_id=1,
            )
            # Flush
            for handler in logger._logger.handlers:
                handler.flush()

            with open(log_path) as fh:
                lines = [l.strip() for l in fh if l.strip()]

            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["event"], "rom_decision")
            self.assertEqual(record["request_id"], "req-1")
            self.assertEqual(record["decision"], "migrate")
            self.assertEqual(record["prompt_len"], 512)
            self.assertEqual(record["failed_worker_id"], 2)
        finally:
            os.unlink(log_path)

    def test_log_recovery_success(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            logger = ROMLogger(log_file=log_path)
            logger.log_recovery(
                request_id="req-2",
                decision="recompute",
                success=True,
                latency_s=0.015,
            )
            for handler in logger._logger.handlers:
                handler.flush()

            with open(log_path) as fh:
                record = json.loads(fh.read().strip())

            self.assertEqual(record["event"], "rom_recovery")
            self.assertTrue(record["success"])
            self.assertAlmostEqual(record["latency_s"], 0.015, places=4)
        finally:
            os.unlink(log_path)

    def test_log_recovery_failure_includes_error(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            logger = ROMLogger(log_file=log_path)
            logger.log_recovery(
                request_id="req-3",
                decision="migrate",
                success=False,
                latency_s=0.001,
                error="RayActorError: worker died",
            )
            for handler in logger._logger.handlers:
                handler.flush()

            with open(log_path) as fh:
                record = json.loads(fh.read().strip())

            self.assertFalse(record["success"])
            self.assertIn("RayActorError", record["error"])
        finally:
            os.unlink(log_path)

    def test_log_worker_failure(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            logger = ROMLogger(log_file=log_path)
            logger.log_worker_failure(
                worker_id=3,
                affected_request_ids=["req-a", "req-b"],
                error_msg="ActorDied",
            )
            for handler in logger._logger.handlers:
                handler.flush()

            with open(log_path) as fh:
                record = json.loads(fh.read().strip())

            self.assertEqual(record["event"], "decode_worker_failure")
            self.assertEqual(record["worker_id"], 3)
            self.assertEqual(record["affected_request_count"], 2)
            self.assertIn("req-a", record["affected_request_ids"])
        finally:
            os.unlink(log_path)

    def test_multiple_records_all_valid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            logger = ROMLogger(log_file=log_path)
            for i in range(5):
                logger.log_decision(
                    request_id=f"req-{i}",
                    prompt_len=128 * (i + 1),
                    bandwidth_mbps=10_000.0,
                    c_mig=0.001 * i,
                    c_recomp=0.010 * i,
                    decision="migrate",
                    policy="rom",
                )
            for handler in logger._logger.handlers:
                handler.flush()

            with open(log_path) as fh:
                lines = [l.strip() for l in fh if l.strip()]

            self.assertEqual(len(lines), 5)
            for line in lines:
                record = json.loads(line)   # must not raise
                self.assertIn("event", record)
        finally:
            os.unlink(log_path)


# ===========================================================================
# LocalScheduler tests  (Ray-free: mock ray.remote)
# ===========================================================================

class TestLocalSchedulerLogic(unittest.TestCase):
    """
    Tests for the LocalScheduler *business logic* extracted from the Ray actor.

    Because spawning a Ray cluster in unit tests is expensive, we test the
    actor's methods by instantiating the class directly (bypassing @ray.remote)
    via a mock of the decorator.
    """

    def _make_scheduler(self, bw_fallback=10_000.0):
        """Instantiate LocalScheduler without Ray."""
        # Temporarily replace ray.remote with identity so the class can be
        # instantiated directly.
        import distserve.rom_local_scheduler as module
        orig_remote = module.ray.remote

        class _NoOp:
            def __call__(self, cls):
                return cls
        module.ray.remote = _NoOp()
        try:
            from distserve.rom_local_scheduler import LocalScheduler
            cfg = ROMConfig(sim_bandwidth_mbps=bw_fallback)
            instance = LocalScheduler.__new__(LocalScheduler)
            LocalScheduler.__init__(instance, cfg)
        finally:
            module.ray.remote = orig_remote
        return instance

    def test_initial_bandwidth_is_fallback(self):
        sched = self._make_scheduler(bw_fallback=5_000.0)
        # Nothing measured yet — should return fallback
        bw = sched.get_bandwidth()
        self.assertEqual(bw, 5_000.0)

    def test_update_bandwidth_ewma(self):
        sched = self._make_scheduler(bw_fallback=10_000.0)
        # Feed one measurement
        sched.update_bandwidth(8_000.0)
        bw = sched.get_bandwidth()
        # EWMA: α=0.2 → 0.2*8000 + 0.8*10000 = 9600
        self.assertAlmostEqual(bw, 9_600.0, delta=1.0)

    def test_update_bandwidth_ignores_nonpositive(self):
        sched = self._make_scheduler(bw_fallback=10_000.0)
        sched.update_bandwidth(-1.0)
        sched.update_bandwidth(0.0)
        self.assertEqual(sched.get_bandwidth(), 10_000.0)

    def test_multiple_bandwidth_updates(self):
        sched = self._make_scheduler(bw_fallback=10_000.0)
        for _ in range(10):
            sched.update_bandwidth(5_000.0)
        bw = sched.get_bandwidth()
        # After many updates with 5000, EWMA should converge toward 5000
        self.assertLess(bw, 10_000.0)
        self.assertGreater(bw, 4_000.0)

    def test_register_and_get_best_worker(self):
        sched = self._make_scheduler()
        sched.register_worker(0)
        sched.register_worker(1)
        sched.register_worker(2)
        sched.update_queue_length(0, 10)
        sched.update_queue_length(1, 3)
        sched.update_queue_length(2, 7)
        best = sched.get_best_decode_worker()
        self.assertEqual(best, 1)

    def test_get_best_worker_with_exclusion(self):
        sched = self._make_scheduler()
        sched.register_worker(0)
        sched.register_worker(1)
        sched.update_queue_length(0, 1)
        sched.update_queue_length(1, 5)
        # Exclude worker 0 (the best) → should return 1
        best = sched.get_best_decode_worker(exclude=[0])
        self.assertEqual(best, 1)

    def test_get_best_worker_no_candidates_returns_none(self):
        sched = self._make_scheduler()
        self.assertIsNone(sched.get_best_decode_worker())

    def test_deregister_worker(self):
        sched = self._make_scheduler()
        sched.register_worker(0)
        sched.register_worker(1)
        sched.update_queue_length(0, 1)
        sched.update_queue_length(1, 2)
        sched.deregister_worker(0)
        best = sched.get_best_decode_worker()
        self.assertEqual(best, 1)

    def test_get_state_structure(self):
        sched = self._make_scheduler(bw_fallback=5_000.0)
        sched.register_worker(0)
        sched.update_queue_length(0, 3)
        state = sched.get_state()
        self.assertIn("bandwidth_mbps", state)
        self.assertIn("queue_lengths", state)
        self.assertIn("bw_fresh", state)
        self.assertIsInstance(state["queue_lengths"], dict)

    def test_reset_clears_state(self):
        sched = self._make_scheduler(bw_fallback=10_000.0)
        sched.update_bandwidth(5_000.0)
        sched.register_worker(0)
        sched.update_queue_length(0, 99)
        sched.reset()
        self.assertEqual(sched.get_bandwidth(), 10_000.0)   # fallback restored
        self.assertIsNone(sched.get_best_decode_worker())

    def test_update_queue_clamps_negative(self):
        sched = self._make_scheduler()
        sched.register_worker(0)
        sched.update_queue_length(0, -5)
        self.assertEqual(sched.get_queue_length(0), 0)


# ===========================================================================
# Integration-style test: decision + logger pipeline
# ===========================================================================

class TestDecisionLoggerPipeline(unittest.TestCase):
    """
    End-to-end: run make_decision then log the result, verify the log record
    contains consistent data.
    """

    def setUp(self):
        reset_python_logger()

    def tearDown(self):
        reset_python_logger()

    def test_decision_then_log(self):
        cfg = ROMConfig(
            policy="rom",
            sim_bandwidth_mbps=100.0,   # slow → recompute expected for long prompts
            num_layers=32, num_kv_heads=32, head_dim=128, dtype_bytes=2,
            prefill_time_table={128: 0.010, 512: 0.032, 1024: 0.060},
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            logger = ROMLogger(log_file=log_path)
            prompt_len = 1024
            bw = 100.0

            decision, c_mig, c_recomp = make_decision(prompt_len, bw, cfg)
            logger.log_decision(
                request_id="integration-1",
                prompt_len=prompt_len,
                bandwidth_mbps=bw,
                c_mig=c_mig,
                c_recomp=c_recomp,
                decision=decision,
                policy=cfg.policy,
                failed_worker_id=0,
                target_worker_id=1,
            )
            for h in logger._logger.handlers:
                h.flush()

            with open(log_path) as fh:
                record = json.loads(fh.read().strip())

            # Decision should be consistent with the logged costs
            if record["c_mig_s"] <= record["c_recomp_s"]:
                self.assertEqual(record["decision"], "migrate")
            else:
                self.assertEqual(record["decision"], "recompute")

            self.assertEqual(record["prompt_len"], prompt_len)
        finally:
            os.unlink(log_path)


if __name__ == "__main__":
    unittest.main()
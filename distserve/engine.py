import time
import copy
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import asyncio
import math
import argparse

import ray
from ray.util.placement_group import PlacementGroup

from distserve.config import (
    ModelConfig, 
    DisaggParallelConfig, 
    ParallelConfig, 
    CacheConfig, 
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.logger import init_logger
from distserve.request import (
    SamplingParams,
    Request,
    create_request,
)
from distserve.tokenizer import get_tokenizer
from distserve.utils import Counter
from distserve.single_stage_engine import (
    StepOutput,
    ContextStageLLMEngine,
    DecodingStageLLMEngine
)
from distserve.lifetime import LifetimeEvent, LifetimeEventType
from distserve.rom_config import ROMConfig
from distserve.rom_decision import make_decision
from distserve.rom_local_scheduler import LocalScheduler
from distserve.rom_logger import ROMLogger

logger = init_logger(__name__)


class LLMEngine:
    """
    LLMEngine: An LLMEngine launches the model executor workers and maintains runtime information.

    ## Overview

    This class, LLMEngine, receives requests from upper wrapper class and provides
    interface LLMEngine.generate() that yields the generated tokens for each request.

    It supports the feature of "disaggregate", which basically means to run 
    the context stage and the decoding stage on different GPUs to avoid interference.

    ## Implementation

    First let's inspect the automaton of one request:

            After
            context
            stage        |-------------|
    Waiting --------> Decoding <-------| After one decoding stage
                         |
                         |
                         V
                      Finished

    This class is implemented based on queues and event loops. There are three
    queues, two for scheduling and one for communication between event loops:
      - The waiting queue, maintained inside the ContextStageScheduler, which
        contains all the requests that are waiting for processing.
      - The decoding queue, maintained inside the DecodingStageScheduler, which
        contains all the requests that need further decoding.
      - The "bridge" queue, which contains all the requests that have just finished
        the context stage but have not been accepted by the decoding stage.
        (Producer: context stage event loop, Consumer: decoding stage event loop)
      
    Two event loops are executed concurrently and endlessly:
      - Context stage event loop. This event loop fetches requests from the waiting
        queue, forwards them to the context stage, and then puts them into the
        "bridge" queue.
      - Decoding stage event loop. This event loop accepts requests from the
        "bridge" queue (put them into the decoding queue), and then fetches requests
        from the decoding queue, forwards them to the decoding stage, and then
        informs the caller of the generated tokens.

    Note: Users may not use LLMEngine directly, but use more user-friendly wrapper classes
    OfflineLLM and AsyncLLM instead.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig
    ):
        self.model_config = model_config
        self.disagg_parallel_config = disagg_parallel_config
        self.cache_config = cache_config
        self.context_sched_config = context_sched_config
        self.decoding_sched_config = decoding_sched_config

        self.request_counter = Counter()
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        self.bridge_queue = asyncio.Queue()
        
        logger.info("Initializing placement group")
        placement_groups = self._init_placement_groups()
        
        logger.info("Initializing context stage LLM engine")
        self.context_engine = ContextStageLLMEngine(
            self.bridge_queue,
            model_config,
            disagg_parallel_config.context,
            cache_config,
            context_sched_config,
            placement_groups,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback
        )
        self._rom_init()
        self.apply_model_config_to_rom(
            num_layers=self.model_config.hf_config.num_hidden_layers,
            num_kv_heads=self.model_config.get_num_heads(),
            head_dim=self.model_config.get_head_size(),
        )
        logger.info("Initializing decoding stage LLM engine")
        self.decoding_engine = DecodingStageLLMEngine(
            self.bridge_queue,
            model_config,
            disagg_parallel_config.decoding,
            cache_config,
            decoding_sched_config,
            placement_groups,
            self.context_engine.clear_migrated_blocks_callback,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback,
            llm_engine=self,
            rom_config=self.rom_config,
        )
       
        # request_id -> list of StepOutput
        # Created when calling self.generate()
        # Cleared when the request is finished
        self.request_outputs: Dict[int, asyncio.Queue[StepOutput]] = {}
        
        # request_id -> list of LifetimeEvent
        # Created when calling self.generate()
        # Cleared by the caller of self.generate() (i.e. the engine does not clear that)
        # TODO: clear this automatically to avoid memory leak
        self.request_lifetime_events: Dict[int, List[LifetimeEvent]] = {}
      
        self.engine_initialized = False
    
    def _rom_init(self) -> None:
        """
        Initialise all RoM scheduler components
        """
        # Load config (respects ROM_* env vars)
        self.rom_config: ROMConfig = ROMConfig.from_env()
    
        # Local scheduler actor — named so the decoding engine can locate it
        self._local_scheduler: LocalScheduler = LocalScheduler.options(
            name="rom_local_scheduler",
            get_if_exists=True,
        ).remote(self.rom_config)
    
        # Structured JSONL logger
        self._rom_logger: ROMLogger = ROMLogger(log_file=self.rom_config.log_file)
    
        # prompt_len map: request_id (str) → prompt token count (int)
        # Populated in generate(); consumed by get_rom_decision()
        self._prompt_len_map: dict[str, int] = {}
    
        self._rom_logger._logger.info(
            '{"event":"rom_init","policy":"%s","monitor_enabled":%s}'
            % (self.rom_config.policy, str(self.rom_config.monitor_enabled).lower())
        )
    
    def _on_new_step_output_callback(self, request_id: int, step_output: StepOutput):
        """
        Called by self.context_engine or self.decoding_engine when a new output token
        is generated
        """
        self.request_outputs[request_id].put_nowait(step_output)
        
    def _on_new_lifetime_event_callback(self, request_id: int, event: LifetimeEvent, dont_add_if_dup: bool = False):
        """
        Called by self.context_engine or self.decoding_engine when a new lifetime event
        is generated
        """
        # if dont_add_if_dup == True and self.request_lifetime_events[request_id][-1].event_type == event.event_type, don't add it
        if dont_add_if_dup and \
            len(self.request_lifetime_events[request_id]) > 0 and \
                self.request_lifetime_events[request_id][-1].event_type == event.event_type:
            return
        self.request_lifetime_events[request_id].append(event)
    
    def _init_placement_groups(self) -> Optional[List[PlacementGroup]]:
        """
        Create placement groups for all engines and all workers
        
        Currently we force the same layer of the context & decoding stage to be executed
        on the same node (we call this "aligned"). This simplifies k/v cache migration.
        """
        context_pp = self.disagg_parallel_config.context.pipeline_parallel_size
        context_tp = self.disagg_parallel_config.context.tensor_parallel_size
        decoding_pp = self.disagg_parallel_config.decoding.pipeline_parallel_size
        decoding_tp = self.disagg_parallel_config.decoding.tensor_parallel_size
        
        # Each placement group is responsible for `layer_per_placement_group` layers
        layer_per_context_pp = self.model_config.get_num_layers(self.disagg_parallel_config.context)
        layer_per_decoding_pp = self.model_config.get_num_layers(self.disagg_parallel_config.decoding)
        layer_per_placement_group = math.lcm(layer_per_context_pp, layer_per_decoding_pp)
        
        # Each placement group contains `workers_per_placement_group` workers
        workers_per_placement_group = \
            layer_per_placement_group // layer_per_context_pp * context_tp \
            + layer_per_placement_group // layer_per_decoding_pp * decoding_tp
        
        # There should be `num_placement_groups` placement groups in total
        num_placement_groups = self.model_config.get_num_layers() // layer_per_placement_group
        assert num_placement_groups * workers_per_placement_group == \
            context_pp * context_tp + decoding_pp * decoding_tp
        
        # Create placement groups
        placement_groups = []
        for i in range(num_placement_groups):
            placement_group = ray.util.placement_group(
                [ { "GPU": 1 }] * workers_per_placement_group,
                strategy="STRICT_PACK",
            )
            ray.get(placement_group.ready(), timeout=1000)
            placement_groups.append(placement_group)
        
        return placement_groups
        
    async def initialize(self):
        await asyncio.gather(
            self.context_engine.initialize(),
            self.decoding_engine.initialize()
        )
        await self.decoding_engine.register_kvcache_mem_handles(
            self.context_engine.parallel_config,
            self.context_engine.kv_cache_mem_handles
        )
        self.decoding_engine._build_worker_index()
        await self._register_rom_workers()
        self.engine_initialized = True
    
    async def _register_rom_workers(self) -> None:
        if not hasattr(self, "_local_scheduler"):
            return
        
        worker_count = len(self.decoding_engine.workers)
        if worker_count == 0:
            return
        tasks = [
            self._local_scheduler.register_worker.remote(worker_id)
            for worker_id in range(worker_count)
        ]
        await asyncio.gather(*tasks)

    def _remote_call_all_workers(
        self, 
        func_name: str, 
        *args
    ):
        """
        call func_name on all workers, blocked until all workers finish, and return all the results
        """
        handlers = self._remote_call_all_workers_async(func_name, *args)
        return ray.get(handlers)

    def _remote_call_all_workers_async(
        self, 
        func_name: str,
        *args
    ):
        """
        call func_name asynchronously on all workers (context/decoding/both), return the futures immediately
        """
        handlers = self.context_engine._remote_call_all_workers_async(func_name, *args)
        handlers += self.decoding_engine._remote_call_all_workers_async(func_name, *args)
        return handlers

    async def _start_my_event_loop(self):
        #pass
        await self._probe_bandwidth()
    
    async def start_all_event_loops(self):
        """
        start_all_event_loops: Start context_engine's, decoding_engine's, and
        mine (LLMEngine's) event loops
        """
        logger.info("Starting LLMEngine's event loops")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."
        await asyncio.gather(
            self.context_engine.start_event_loop(),
            self.decoding_engine.start_event_loop(),
            self._start_my_event_loop()
        )
       
    async def generate(
        self,
        prompt: Optional[str],
        prompt_len: Optional[int],
        prompt_token_ids: Optional[List[str]],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
        request_id: Optional[int] = None,
    ) -> AsyncGenerator[StepOutput, None]:
        """
        generate - Generate outputs for one request
        
        This function is intended to be used as an async generator, i.e., it can be
        used in a for loop. For example, `async for output in engine.generate(...)`
        """
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before generating."
        req = create_request(
            prompt,
            prompt_len,
            prompt_token_ids,
            sampling_params,
            self.request_counter,
            self.tokenizer,
            arrival_time,
            request_id,
        )
        self.request_outputs[req.request_id] = asyncio.Queue()
        self.request_lifetime_events[req.request_id] = []
        
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Issued))
        self.context_engine.add_request(req)
        prompt_len_value = prompt_len
        if prompt_len_value is None:
            if prompt_token_ids is not None:
                prompt_len_value = len(prompt_token_ids)
            elif prompt is not None:
                prompt_len_value = len(self.tokenizer.tokenize(prompt))
            else:
                prompt_len_value = 0
        self._prompt_len_map[request_id] = prompt_len_value
        
        while True:
            try:
                step_output = await self.request_outputs[req.request_id].get()
            except asyncio.CancelledError:
                # The engine returns
                # Exception should be handled by the engine, not me
                return
            except GeneratorExit:
                return
            yield step_output
            if step_output.is_finished:
                break
                
        del self.request_outputs[req.request_id]
        self._prompt_len_map.pop(req.request_id, None) 

    def abort_request(self, request_id: int):
        self.context_engine.abort_request(request_id)
        self.decoding_engine.abort_request(request_id)
    
    async def get_rom_decision(
        self,
        request_id: str,
        bandwidth_mbps_override: float | None = None,
    ) -> tuple[str, int, float, float]:
        """
        Compute and return the RoM decision for request_id.
    
        Called by the failure monitor in DecodingStageLLMEngine when a decode
        worker fails and a recovery path must be chosen.
    
        Parameters
        ----------
        request_id             : the failing request's unique id
        bandwidth_mbps_override: if provided, use this bandwidth estimate
                                instead of querying LocalScheduler (useful
                                for tests or when the caller already holds
                                a fresh measurement).
    
        Returns
        -------
        (decision, prompt_len, c_mig, c_recomp)
            decision   : "migrate" or "recompute"
            prompt_len : number of tokens in the original prompt
            c_mig      : estimated migration cost (seconds)
            c_recomp   : estimated recomputation cost (seconds)
    
        Notes
        -----
        If request_id is unknown, prompt_len
        defaults to 0 and the decision defaults to "migrate".
        """
        prompt_len = self._prompt_len_map.get(request_id, 0)
    
        if bandwidth_mbps_override is not None:
            bw = bandwidth_mbps_override
        else:
            bw = await self._local_scheduler.get_bandwidth.remote()
    
        decision, c_mig, c_recomp = make_decision(
            prompt_len=prompt_len,
            bandwidth_mbps=bw,
            rom_config=self.rom_config,
        )
        return decision, prompt_len, c_mig, c_recomp

    async def _probe_bandwidth(self) -> None:
        """
        Probe inter-node bandwidth and push to scheduler.
    
        Runs once per monitor_interval_s (same cadence as the health monitor).
        Uses a small Ray object-store round-trip to approximate NIC throughput.
        """
        import time as _time
    
        PROBE_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB probe payload
        PROBE_INTERVAL_S = self.rom_config.monitor_interval_s * 5  # less frequent
    
        while True:
            try:
                payload = b"\x00" * PROBE_SIZE_BYTES
                t0 = _time.monotonic()
                ref = ray.put(payload)
                await asyncio.get_event_loop().run_in_executor(None, ray.get, ref)
                elapsed = _time.monotonic() - t0
                if elapsed > 0:
                    bw_mbps = (PROBE_SIZE_BYTES / elapsed) / 1_000_000
                    self._local_scheduler.update_bandwidth.remote(bw_mbps)
            except Exception:
                pass  # probe failure is non-fatal
            await asyncio.sleep(PROBE_INTERVAL_S)

    def apply_model_config_to_rom(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """
        Override ROMConfig architecture parameters with values extracted from
        the loaded model.
    
        Call this inside LLMEngine.__init__() after _remote_call_all_workers()
        returns the architecture metadata:
    
            model_meta = ray.get(
                self.context_engine.workers[0][0].get_model_config.remote()
            )
            self.apply_model_config_to_rom(
                num_layers   = model_meta["num_layers"],
                num_kv_heads = model_meta["num_kv_heads"],
                head_dim     = model_meta["head_dim"],
            )
    
        Parameters
        ----------
        num_layers   : number of transformer layers
        num_kv_heads : number of K/V attention heads per layer
        head_dim     : feature dimension of each head
        """
        self.rom_config.update_from_model_config(num_layers, num_kv_heads, head_dim)
        self._rom_logger._logger.info(
            '{"event":"rom_model_config","num_layers":%d,"num_kv_heads":%d,'
            '"head_dim":%d}' % (num_layers, num_kv_heads, head_dim)
        )
        
def add_engine_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--context-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-tensor-parallel-size", type=int, default=1)
    
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=16)
    
    parser.add_argument("--context-sched-policy", type=str, default="fcfs")
    parser.add_argument("--context-max-batch-size", type=int, default=256)
    parser.add_argument("--context-max-tokens-per-batch", type=int, default=4096)
    
    parser.add_argument("--decoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--decoding-max-batch-size", type=int, default=256)
    parser.add_argument("--decoding-max-tokens-per-batch", type=int, default=8192)
    
    parser.add_argument("--simulator-mode", action="store_true")
    parser.add_argument("--profiler-data-path", type=str, default=None)
    parser.add_argument("--gpu-mem-size-gb", type=float, default=None)
    
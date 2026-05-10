"""
Usage example:

python -m distserve.api_server.distserve_api_server \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer {args.model} \\
    \\
    --context-tensor-parallel-size {context_tp} \\
    --context-pipeline-parallel-size {context_pp} \\
    --decoding-tensor-parallel-size {decoding_tp} \\
    --decoding-pipeline-parallel-size {decoding_pp} \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 128 \\
    --gpu-memory-utilization 0.95 \\
    --swap-space 16 \\
    \\
    --context-sched-policy fcfs \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 8192 \\
    \\
    --decoding-sched-policy fcfs \\
    --decoding-max-batch-size 1024 \\
    --decoding-max-tokens-per-batch 65536
"""

import argparse
import json
from typing import AsyncGenerator, List, Tuple, Optional
import asyncio
import time
import traceback
import sys, os
import signal
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

import distserve
import distserve.engine
from distserve.llm import AsyncLLM
from distserve.request import SamplingParams
from distserve.utils import random_uuid, set_random_seed
from distserve.logger import init_logger
from distserve.single_stage_engine import StepOutput
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.lifetime import json_encode_lifetime_events

import ray

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine: Optional[AsyncLLM] = None


class ROMDecisionResponse(BaseModel):
    request_id: str
    prompt_len: int
    bandwidth_mbps: float
    c_mig_s: float
    c_recomp_s: float
    decision: str
    policy: str


class ROMRecoverRequest(BaseModel):
    request_id: str
    force: Optional[str] = None


class ROMRecoverResponse(BaseModel):
    request_id: str
    decision: str
    triggered: bool
    message: str


def _require_engine() -> AsyncLLM:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized",
        )
    return engine


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    logger.info("Received a request.")
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    llm = _require_engine()
    results_generator = llm.generate(
        request_id, prompt=prompt, sampling_params=sampling_params
    )

    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for step_output in results_generator:
                text_output = step_output.request.get_response()
                ret = {"text": text_output}
                yield (json.dumps(ret)).encode("utf-8")

        async def abort_request() -> None:
            await llm.abort(request_id)

        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        # Currently we do not support request abortion, so we comment this line.
        # TODO implement request abortion.
        # background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        final_outputs: List[Tuple[StepOutput, float]] = []   # (step_output, timestamp)
        async for step_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await llm.abort(request_id)
                return Response(status_code=499)
            final_outputs.append((step_output, time.perf_counter()))

        request_events = llm.get_and_pop_request_lifetime_events(request_id)
        text_output = prompt + ''.join([step_output[0].new_token for step_output in final_outputs])
        ret = {
            "text": text_output,
            "timestamps": [step_output[1] for step_output in final_outputs],
            "lifetime_events": json_encode_lifetime_events(request_events)
        }
        return JSONResponse(ret)


@app.get("/v1/rom/decision")
async def rom_decision(
    request_id: str,
    bandwidth_mbps: Optional[float] = None,
) -> ROMDecisionResponse:
    """
    Return the ROM decision for an in-flight request without acting on it.
 
    Query parameters
    ----------------
    request_id     : (required) the request to evaluate
    bandwidth_mbps : (optional) override the bandwidth used in the cost model
 
    Returns 404 if the request is not currently in flight.
    """
    llm_engine = _require_engine().engine

    if request_id not in llm_engine._prompt_len_map:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request_id!r} not found in active request map",
        )
 
    decision, prompt_len, c_mig, c_recomp = await llm_engine.get_rom_decision(
        request_id=request_id,
        bandwidth_mbps_override=bandwidth_mbps,
    )
 
    return ROMDecisionResponse(
        request_id=request_id,
        prompt_len=prompt_len,
        bandwidth_mbps=bandwidth_mbps or 0.0,
        c_mig_s=c_mig,
        c_recomp_s=c_recomp,
        decision=decision,
        policy=llm_engine.rom_config.policy,
    )


@app.post("/v1/rom/recover")
async def rom_recover(body: ROMRecoverRequest) -> ROMRecoverResponse:
    """
    Manually trigger the RoM recovery path for a specific request.
 
    Body
    ----
    request_id : the request to recover
    force      : "migrate", "recompute", or null (uses the ROM policy)
 
    Returns 404 if the request is not found.
    Returns 400 if ``force`` is an invalid value.
 
    NOTE: This endpoint is for TESTING and OPERATIONAL INTERVENTION only.
          In normal operation, recovery is triggered automatically by the
          built-in decode-worker health monitor.
    """
    if body.force and body.force not in {"migrate", "recompute"}:
        raise HTTPException(
            status_code=400,
            detail=f"force must be 'migrate', 'recompute', or null",
        )
 
    llm_engine = _require_engine().engine
 
    if body.request_id not in llm_engine._prompt_len_map:
        raise HTTPException(
            status_code=404,
            detail=f"Request {body.request_id!r} not found",
        )
 
    # Determine decision
    if body.force:
        decision = body.force
        c_mig = c_recomp = 0.0
    else:
        decision, _pl, c_mig, c_recomp = await llm_engine.get_rom_decision(
            body.request_id
        )
 
    # Execute recovery via the decoding engine
    decode_engine = llm_engine.decoding_engine
    try:
        if decision == "migrate":
            import ray as _ray
            local_sched = _ray.get_actor("rom_local_scheduler")
            target = await local_sched.get_best_decode_worker.remote()
            await decode_engine.recover_via_migrate(body.request_id, target)
        else:
            await decode_engine.recover_via_recompute(body.request_id)
        triggered = True
        message = f"Recovery ({decision}) triggered successfully"
    except Exception as exc:
        triggered = False
        message = str(exc)
 
    return ROMRecoverResponse(
        request_id=body.request_id,
        decision=decision,
        triggered=triggered,
        message=message,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    distserve.engine.add_engine_cli_args(parser)
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    ray.init()
    
    engine = AsyncLLM.from_engine_args(args)

    uvicorn_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    async def main_coroutine():
        task2 = asyncio.create_task(uvicorn_server.serve())
        
        async def start_event_loop_wrapper():
            try:
                task = asyncio.create_task(engine.start_event_loop())
                await task
            except Exception as e:
                traceback.print_exc()
                task2.cancel()
                os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
        
        task1 = asyncio.create_task(start_event_loop_wrapper())
        
        try:
            await task2
        except:
            # This is a workaround
            # When task1 exited for some reason (e.g. error in the engine),
            # task2 will raise many exceptions, which is annoying and I do 
            # not know why
            pass
    
    asyncio.run(main_coroutine())
    
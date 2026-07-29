"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import dataclasses
import json
import logging
import os
import time
import random
import urllib
from pathlib import Path
from itertools import chain
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.utils import PDRegistryRequest
from sglang.srt.utils import maybe_wrap_ipv6_address

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error

DEFAULT_DRAIN_STATE_FILE = "/tmp/afd_runtime_state.json"
DEFAULT_DRAIN_WAIT_TIMEOUT_S = 600.0


def setup_logger():
    logger = logging.getLogger("pdlb")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class MiniLoadBalancer:
    def __init__(
        self,
        prefill_configs: List[PrefillConfig],
        decode_servers: List[str],
        drain_state_file: Optional[str] = DEFAULT_DRAIN_STATE_FILE,
        drain_wait_timeout_s: float = DEFAULT_DRAIN_WAIT_TIMEOUT_S,
    ):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers
        self.drain_state_file = drain_state_file
        self.drain_wait_timeout_s = max(0.0, float(drain_wait_timeout_s))
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._inflight_requests = 0
        self._forwarded_requests = 0
        self._forwarded_prompts = 0
        self._stats_window_start = time.time()
        self._metrics_output_file = None
        if _env_bool("SGLANG_ENABLE_AFD_METRICS"):
            metrics_dir = Path(os.getenv("SGLANG_AFD_METRICS_DIR", "/tmp/sglang_afd_metrics"))
            metrics_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_output_file = metrics_dir / "afd_metrics_lb.jsonl"

    async def session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=0,
                    limit_per_host=0,
                    force_close=False,
                    enable_cleanup_closed=True,
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=3600),
                )
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _record_forward_start(self, modified_request) -> int:
        batch_size = _get_request_batch_size(modified_request) or 1
        self._inflight_requests += 1
        self._forwarded_requests += 1
        self._forwarded_prompts += batch_size
        self._maybe_log_stats()
        return batch_size

    def _record_forward_done(self):
        self._inflight_requests = max(0, self._inflight_requests - 1)

    def _maybe_log_stats(self):
        now = time.time()
        elapsed = now - self._stats_window_start
        if elapsed < 1.0:
            return
        request_rate = self._forwarded_requests / elapsed
        prompt_rate = self._forwarded_prompts / elapsed
        logger.info(
            "forward stats: inflight=%d http_req/s=%.2f prompt/s=%.2f "
            "window_http_req=%d window_prompts=%d",
            self._inflight_requests,
            request_rate,
            prompt_rate,
            self._forwarded_requests,
            self._forwarded_prompts,
        )
        self._write_metrics_sample(now, elapsed, request_rate, prompt_rate)
        self._forwarded_requests = 0
        self._forwarded_prompts = 0
        self._stats_window_start = now

    def _write_metrics_sample(
        self,
        timestamp: float,
        elapsed_s: float,
        request_rate: float,
        prompt_rate: float,
    ):
        if self._metrics_output_file is None:
            return
        sample = {
            "timestamp": timestamp,
            "worker_type": "lb",
            "arrival_rate_rps": prompt_rate,
            "window_s": elapsed_s,
            "window_requests": self._forwarded_prompts,
        }
        try:
            with open(self._metrics_output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception:
            logger.exception("Failed to write AFD LB metrics to %s", self._metrics_output_file)

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(new_decode_server)

    def select_pair(self):
        # TODO: return some message instead of panic
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        prefill_config = random.choice(self.prefill_configs)
        decode_server = random.choice(self.decode_servers)
        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    def _runtime_state(self) -> str:
        if not self.drain_state_file:
            return "serving"
        path = Path(self.drain_state_file)
        if not path.exists():
            return "serving"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.exception("Failed to read drain state file %s", path)
            return "serving"
        return str(data.get("state", "serving"))

    async def wait_until_serving(self):
        if not self.drain_state_file:
            return
        deadline = time.time() + self.drain_wait_timeout_s
        logged = False
        while self._runtime_state() == "draining":
            if not logged:
                logger.info(
                    "Runtime is draining; waiting before forwarding new request "
                    "(state_file=%s)",
                    self.drain_state_file,
                )
                logged = True
            if time.time() >= deadline:
                raise HTTPException(
                    status_code=503,
                    detail="Runtime is still draining; request wait timed out.",
                )
            await asyncio.sleep(1.0)

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        self._record_forward_start(modified_request)
        prefill_response = None
        decode_response = None
        try:
            session = await self.session()
            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                session.post(f"{decode_server}/{endpoint}", json=modified_request),
            ]

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if "return_logprob" in modified_request:

                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await decode_response.json()

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )
        finally:
            if prefill_response is not None:
                prefill_response.release()
            if decode_response is not None:
                decode_response.release()
            self._record_forward_done()

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            self._record_forward_start(modified_request)
            prefill_response = None
            decode_response = None
            try:
                session = await self.session()
                # Create the tasks for both prefill and decode requests
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                    session.post(f"{decode_server}/{endpoint}", json=modified_request),
                ]
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    first_prefill_chunk = (
                        prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    prefill_response.release()
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk
            finally:
                if prefill_response is not None:
                    prefill_response.release()
                if decode_response is not None:
                    decode_response.release()
                self._record_forward_done()

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
load_balancer: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.on_event("shutdown")
async def shutdown():
    if load_balancer is not None:
        await load_balancer.close()


@app.get("/health_generate")
async def health_check():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    session = await load_balancer.session()
    # Create the tasks
    tasks = []
    for server in chain(prefill_servers, decode_servers):
        tasks.append(session.post(f"{server}/health_generate"))
    for i, response in enumerate(asyncio.as_completed(tasks)):
        resp = await response
        resp.release()
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    session = await load_balancer.session()
    # Create the tasks
    tasks = []
    for server in chain(prefill_servers, decode_servers):
        tasks.append(session.post(f"{server}/flush_cache"))
    for i, response in enumerate(asyncio.as_completed(tasks)):
        resp = await response
        resp.release()
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    session = await load_balancer.session()
    for server in chain(prefill_servers):
        server_info = await session.get(f"{server}/get_server_info")
        try:
            prefill_infos.append(await server_info.json())
        finally:
            server_info.release()
    for server in chain(decode_servers):
        server_info = await session.get(f"{server}/get_server_info")
        try:
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])
        finally:
            server_info.release()

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


@app.get("/get_model_info")
async def get_model_info():
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    await load_balancer.wait_until_serving()
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await load_balancer.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    await load_balancer.wait_until_serving()
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    session = await load_balancer.session()
    try:
        response = await session.get(f"{prefill_server}/v1/models")
        if response.status != 200:
            raise HTTPException(
                status_code=response.status,
                detail=f"Prefill server error: Status {response.status}",
            )
        return ORJSONResponse(content=await response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "response" in locals():
            response.release()


@app.post("/register")
async def register(obj: PDRegistryRequest):
    if obj.mode == "prefill":
        load_balancer.add_prefill_server(
            PrefillConfig(obj.registry_url, obj.bootstrap_port)
        )
        logger.info(
            f"Registered prefill server: {obj.registry_url} with bootstrap port: {obj.bootstrap_port}"
        )
    elif obj.mode == "decode":
        load_balancer.add_decode_server(obj.registry_url)
        logger.info(f"Registered decode server: {obj.registry_url}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be either PREFILL or DECODE.",
        )

    logger.info(
        f"#Prefill servers: {len(load_balancer.prefill_configs)}, "
        f"#Decode servers: {len(load_balancer.decode_servers)}"
    )

    return Response(status_code=200)


def run(
    prefill_configs,
    decode_addrs,
    host,
    port,
    drain_state_file=DEFAULT_DRAIN_STATE_FILE,
    drain_wait_timeout_s=DEFAULT_DRAIN_WAIT_TIMEOUT_S,
):
    global load_balancer
    load_balancer = MiniLoadBalancer(
        prefill_configs,
        decode_addrs,
        drain_state_file=drain_state_file,
        drain_wait_timeout_s=drain_wait_timeout_s,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # FIXME: remove this, use the unified entry point: sglang.srt.disaggregation.launch_lb
    from sglang.srt.disaggregation.launch_lb import main

    main()

import asyncio
import json
import math
import os
import signal
import subprocess
import time
from typing import List, Optional, Tuple, Dict, Any

import aiohttp
import requests


def start_vllm(
    model_id: str,
    port: int = 8080,
    stdout_path: Optional[str] = None,
    tp_size: int = 1,
    max_seq_len: Optional[int] = None,
    wait: bool = True
) -> Tuple[int, str]:
    """Start a vllm server in the background and return its PID and log path.

    The server is started detached from the current process (using a new session).
    If `stdout_path` is not provided a timestamped log file is created under
    `.local/` in the current working directory. stderr is redirected to the same
    file to consolidate logging output.

    Returns:
        Tuple containing (PID, stdout_log_path).
    """
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--port",
        str(port),
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--disable-cascade-attn",
        "--async-scheduling",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        str(max_seq_len) if max_seq_len is not None else "16384",
    ]

    session_suffix = f"{int(time.time() * 1000)}"

    if stdout_path is None:
        stdout_path = os.path.join(
            os.getcwd(), f".local/vllm_stdout_{port}_{session_suffix}.log"
        )

    def _open_log(path: str):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return open(path, "a")

    stdout = _open_log(stdout_path)
    stderr = stdout

    # Start process detached from this terminal (Linux/Unix).
    # preexec_fn=os.setsid makes the child the leader of a new session so it
    # won't be killed when the parent exits.
    try:
        proc = subprocess.Popen(
            cmd, stdout=stdout, stderr=stderr, preexec_fn=os.setsid
        )
    finally:
        stdout.close()
        if stderr is not stdout:
            stderr.close()
    if wait:
        wait_for_server(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-llm-simulator",
            timeout=300.0,
            poll_interval=10.0,
        )
    return proc.pid, stdout_path

def _build_token_like_prompt(token_count: int) -> str:
    """Construct a deterministic pseudo-token prompt of the requested length."""
    base_tokens = [
        "a",
    ]
    if token_count <= 0:
        return ""
    repeated = base_tokens * token_count
    return " ".join(repeated)


def benchmark_openai_compatible_server(
    model_id: str,
    input_prompt_len: int,
    output_len: int,
    base_url: str,
    api_key: Optional[str] = None,
    request_timeout: int = 120,
) -> Dict[str, Any]:
    """Measure prefill and per-token decode latency for an OpenAI-compatible server.

    Args:
        model_id: Identifier understood by the remote server.
        input_prompt_len: Approximate prompt length (in tokens) to send.
        output_len: Maximum number of completion tokens to request.
        base_url: Base REST endpoint, defaults to http://localhost:8080/v1.
        api_key: API key for authorization. Falls back to OPENAI_API_KEY env var.
        request_timeout: Total request timeout in seconds.

    Returns:
        Tuple containing (prefill_time_seconds, decode_time_per_token_seconds).

    Raises:
        RuntimeError: If dependencies are missing or the server response is invalid.
    """
    if input_prompt_len < 0:
        raise ValueError("input_prompt_len must be non-negative")
    if output_len <= 0:
        raise ValueError("output_len must be positive")

    effective_base = base_url
    effective_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not effective_key:
        raise RuntimeError("Missing API key. Provide api_key or set OPENAI_API_KEY.")

    payload = {
        "model": model_id,
        "prompt": _build_token_like_prompt(input_prompt_len),
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    headers = {
        "Authorization": f"Bearer {effective_key}",
        "Content-Type": "application/json",
    }

    url = f"{effective_base}/completions"

    start_ts = time.perf_counter()
    first_token_ts: Optional[float] = None
    last_event_ts: Optional[float] = None
    completion_tokens: Optional[int] = None

    with requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
        stream=True,
    ) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body_preview = response.text[:500]
            raise RuntimeError(f"Remote server error: {body_preview}") from exc

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            if raw_line.startswith(b"data: "):
                data_bytes = raw_line[6:]
            else:
                continue

            if data_bytes.strip() == b"[DONE]":
                break

            try:
                payload_chunk = json.loads(data_bytes)
            except json.JSONDecodeError:
                continue

            event_ts = time.perf_counter()
            last_event_ts = event_ts

            choices = payload_chunk.get("choices", [])
            if choices:
                delta = choices[0]
                text_piece = delta.get("content") or delta.get("text")
                if text_piece and first_token_ts is None:
                    first_token_ts = event_ts

            usage = payload_chunk.get("usage")
            if usage and "completion_tokens" in usage:
                completion_tokens = usage["completion_tokens"]

    if first_token_ts is None:
        raise RuntimeError("Did not receive any completion tokens from the server.")

    prefill_time = first_token_ts - start_ts

    if completion_tokens is None or completion_tokens <= 0:
        # If the server did not return usage, approximate from event count.
        completion_tokens = 1

    if last_event_ts is None:
        last_event_ts = time.perf_counter()

    decode_duration = max(0.0, last_event_ts - first_token_ts)
    decode_time_per_token = decode_duration / max(1, completion_tokens)
    
    # Calculate average time between tokens (TBT)
    # If we have completion_tokens tokens, there are (completion_tokens - 1) intervals if > 1
    # If completion_tokens == 1, TBT is 0 or undefined. Let's say 0.
    # Actually, decode_duration covers (completion_tokens) tokens arriving? 
    # Usually decode starts after first token.
    # first_token_ts is when first token arrived.
    # last_event_ts is when last token arrived.
    # So the duration covers (completion_tokens - 1) intervals.
    
    avg_tbt = 0.0
    if completion_tokens > 1:
        avg_tbt = decode_duration / (completion_tokens - 1)

    return {
        "latency": last_event_ts - start_ts,
        "ttft": prefill_time,
        "decode_time_per_token": decode_time_per_token,
        "completion_tokens": completion_tokens,
        "avg_tbt": avg_tbt,
        "input_len": input_prompt_len,
        "output_len": output_len
    }


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def stop_vllm(pid: int, timeout: float = 5.0) -> None:
    """Gracefully terminate the vLLM process (and its process group) if running."""

    if pid <= 0:
        return

    def _send_signal(signal_type: int) -> bool:
        try:
            if hasattr(os, "killpg"):
                os.killpg(pid, signal_type)
            else:
                os.kill(pid, signal_type)
            return True
        except ProcessLookupError:
            return False

    if not _process_alive(pid):
        return

    if _send_signal(signal.SIGTERM) and timeout > 0:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _process_alive(pid):
                return
            time.sleep(0.1)

    if _process_alive(pid):
        _send_signal(signal.SIGKILL)

def wait_for_server(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> None:
    """Block until the OpenAI-compatible endpoint responds to a /models request."""

    effective_base = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    deadline = time.time() + timeout
    models_endpoint = f"{effective_base}/models"

    last_error: Optional[str] = None
    while time.time() < deadline:
        try:
            response = requests.get(
                models_endpoint, headers=headers, timeout=poll_interval
            )
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code} body={response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        print(f"Waiting for server... /deadline in {deadline - time.time():.1f}s")
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Server at {models_endpoint} did not become ready within {timeout}s. Last error: {last_error}"
    )


def benchmark_openai_compatible_server_with_stats(
    model_id: str,
    input_prompt_len: int,
    output_len: int,
    base_url: str,
    api_key: Optional[str] = None,
    request_timeout: int = 120,
    iterations: int = 8,
    warmup_iterations: int = 3,
) -> dict:
    """Run multiple benchmark iterations and compute statistics.

    Args:
        model_id: Identifier understood by the remote server.
        input_prompt_len: Approximate prompt length (in tokens) to send.
        output_len: Maximum number of completion tokens to request.
        base_url: Base REST endpoint, defaults to http://localhost:8080/v1.
        api_key: API key for authorization. Falls back to OPENAI_API_KEY env var.
        request_timeout: Total request timeout in seconds.
        iterations: Number of benchmark iterations to run.
        warmup_iterations: Number of warmup iterations to discard from statistics.

    Returns:
        Dictionary containing detailed statistics including averages and standard deviations.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if warmup_iterations >= iterations:
        raise ValueError("warmup_iterations must be less than total iterations")

    prefill_times: List[float] = []
    decode_times: List[float] = []
    detailed_results: List[Dict[str, Any]] = []

    print(f"Running benchmark with {iterations} iterations ({warmup_iterations} warmup)...")

    for i in range(iterations):
        try:
            result = benchmark_openai_compatible_server(
                model_id=model_id,
                input_prompt_len=input_prompt_len,
                output_len=output_len,
                base_url=base_url,
                api_key=api_key,
                request_timeout=request_timeout,
            )
            
            # Extract metrics for legacy stats calculation
            prefill_time = result["ttft"]
            decode_time = result["decode_time_per_token"]

            if i >= warmup_iterations:
                prefill_times.append(prefill_time)
                decode_times.append(decode_time)
                detailed_results.append(result)
                print(f"  Iteration {i - warmup_iterations + 1}/{iterations - warmup_iterations}: "
                      f"prefill={prefill_time:.4f}s, decode={decode_time:.6f}s/token")
            else:
                print(f"  Warmup {i + 1}/{warmup_iterations}: "
                      f"prefill={prefill_time:.4f}s, decode={decode_time:.6f}s/token")

        except Exception as e:
            print(f"  Iteration {i + 1} failed: {e}")
            # Continue with other iterations, but note the failure
            continue

    if not prefill_times:
        raise RuntimeError("All benchmark iterations failed")

    # Compute statistics
    prefill_avg = sum(prefill_times) / len(prefill_times)
    decode_avg = sum(decode_times) / len(decode_times)

    prefill_std = math.sqrt(sum(t - prefill_avg for t in prefill_times) ** 2 / len(prefill_times))
    decode_std = math.sqrt(sum((t - decode_avg) ** 2 for t in decode_times) / len(decode_times))

    # Compute additional statistics
    prefill_min = min(prefill_times)
    prefill_max = max(prefill_times)
    decode_min = min(decode_times)
    decode_max = max(decode_times)

    # Compute coefficient of variation (CV = std/mean)
    prefill_cv = prefill_std / prefill_avg if prefill_avg > 0 else 0
    decode_cv = decode_std / decode_avg if decode_avg > 0 else 0

    results = {
        "model_id": model_id,
        "input_prompt_len": input_prompt_len,
        "output_len": output_len,
        "iterations_completed": len(prefill_times),
        "total_iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "prefill": {
            "mean_seconds": prefill_avg,
            "std_seconds": prefill_std,
            "min_seconds": prefill_min,
            "max_seconds": prefill_max,
            "coefficient_of_variation": prefill_cv,
            "all_times": prefill_times,
        },
        "decode": {
            "mean_seconds_per_token": decode_avg,
            "std_seconds_per_token": decode_std,
            "min_seconds_per_token": decode_min,
            "max_seconds_per_token": decode_max,
            "coefficient_of_variation": decode_cv,
            "all_times": decode_times,
        },
        "summary": {
            "avg_prefill_time_ms": prefill_avg * 1000,
            "std_prefill_time_ms": prefill_std * 1000,
            "avg_decode_time_ms_per_token": decode_avg * 1000,
            "std_decode_time_ms_per_token": decode_std * 1000,
            "tokens_per_second": 1.0 / decode_avg if decode_avg > 0 else 0,
        },
        "detailed_results": detailed_results
    }

    return results

async def send_single_request(
    session: aiohttp.ClientSession,
    model_id: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    base_url: str,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Send a single streaming request to the vLLM server and return detailed metrics."""
    start_time = time.perf_counter()
    start_ts = time.time()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    token_times: List[float] = []
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None

    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True
    }

    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)  # Configurable timeout
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if not line:
                        continue

                    line_str = line.decode('utf-8').strip()
                    if not line_str.startswith('data: '):
                        continue

                    data_str = line_str[6:]  # Remove 'data: ' prefix

                    if data_str == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    current_time = time.perf_counter()

                    # Check for token content
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0]
                        text_piece = delta.get("text", "")
                        if text_piece:
                            if first_token_time is None:
                                first_token_time = current_time
                            else:
                                token_times.append(current_time)
                            last_token_time = current_time

                    # Check for usage information
                    usage = chunk.get("usage")
                    if usage:
                        if "completion_tokens" in usage:
                            completion_tokens = usage["completion_tokens"]
                        if "prompt_tokens" in usage:
                            prompt_tokens = usage["prompt_tokens"]

                end_time = time.perf_counter()

                # Calculate metrics
                if first_token_time is None:
                    # No tokens were generated
                    return {
                        "request_id": request_id,
                        "success": False,
                        "latency": end_time - start_time,
                        "error": "No tokens generated",
                        "prompt_tokens": prompt_tokens or 0,
                        "completion_tokens": 0,
                        "total_tokens": prompt_tokens or 0,
                        "time_per_completion_token_seconds": 0,
                        "time_per_total_token_seconds": 0,
                        "time_to_first_token_seconds": 0,
                        "time_between_tokens_seconds": [],
                        "avg_time_between_tokens_seconds": 0,
                        "prefill_time_seconds": 0,
                        "decode_time_seconds": 0
                    }

                # Calculate time metrics
                total_latency = end_time - start_time
                prefill_time = first_token_time - start_time
                decode_time = (last_token_time or first_token_time) - first_token_time

                # Token counts
                completion_tokens = completion_tokens or 0
                total_tokens = (prompt_tokens or 0) + completion_tokens

                # Time between tokens
                time_between_tokens = []
                if len(token_times) > 1:
                    for i in range(1, len(token_times)):
                        time_between_tokens.append(token_times[i] - token_times[i-1])
                elif len(token_times) == 1 and first_token_time and last_token_time:
                    time_between_tokens = [last_token_time - first_token_time]

                avg_time_between_tokens = sum(time_between_tokens) / len(time_between_tokens) if time_between_tokens else 0

                # Time per token calculations
                time_per_completion_token = total_latency / completion_tokens if completion_tokens > 0 else 0
                time_per_total_token = total_latency / total_tokens if total_tokens > 0 else 0

                return {
                    "request_id": request_id,
                    "success": True,
                    "latency": total_latency,
                    "start_time": start_time,
                    "end_time": end_time,
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "time_per_completion_token_seconds": time_per_completion_token,
                    "time_per_total_token_seconds": time_per_total_token,
                    "time_to_first_token_seconds": prefill_time,
                    "time_between_tokens_seconds": time_between_tokens,
                    "avg_time_between_tokens_seconds": avg_time_between_tokens,
                    "prefill_time_seconds": prefill_time,
                    "decode_time_seconds": decode_time,
                    "start_ts": start_ts,
                    "end_ts": time.time()
                }
            else:
                end_time = time.perf_counter()
                error_text = await response.text()
                return {
                    "request_id": request_id,
                    "success": False,
                    "latency": end_time - start_time,
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": f"HTTP {response.status}: {error_text}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "time_per_completion_token_seconds": 0,
                    "time_per_total_token_seconds": 0,
                    "time_to_first_token_seconds": 0,
                    "time_between_tokens_seconds": [],
                    "avg_time_between_tokens_seconds": 0,
                    "prefill_time_seconds": 0,
                    "decode_time_seconds": 0,
                    "start_ts": start_ts,
                    "end_ts": time.time()
                }
    except asyncio.TimeoutError:
        end_time = time.perf_counter()
        return {
            "request_id": request_id,
            "success": False,
            "latency": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time,
            "error": "Request timeout",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "time_per_completion_token_seconds": 0,
            "time_per_total_token_seconds": 0,
            "time_to_first_token_seconds": 0,
            "time_between_tokens_seconds": [],
            "avg_time_between_tokens_seconds": 0,
            "prefill_time_seconds": 0,
            "decode_time_seconds": 0,
            "start_ts": start_ts,
            "end_ts": time.time()
        }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "request_id": request_id,
            "success": False,
            "latency": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time,
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "time_per_completion_token_seconds": 0,
            "time_per_total_token_seconds": 0,
            "time_to_first_token_seconds": 0,
            "time_between_tokens_seconds": [],
            "avg_time_between_tokens_seconds": 0,
            "prefill_time_seconds": 0,
            "decode_time_seconds": 0,
            "start_ts": start_ts,
            "end_ts": time.time()
        }


def _generate_prompt(target_length: int) -> str:
    """Generate a prompt of approximately target_length tokens."""
    # Simple approach: repeat a common sentence pattern
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    sentences_needed = max(1, target_length // 9)  # Approximate 9 tokens per sentence

    prompt = ""
    for i in range(sentences_needed):
        prompt += f"{base_sentence}Sentence {i+1}: "

    return prompt.strip()


async def bench_throughput(
        model_id: str,
        input_prompt_len: int,
        output_len: int,
        base_url: str,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
    """Measure throughput for an OpenAI-compatible server by sending concurrent requests.

    Args:
        model_id: The model identifier to use for requests
        input_prompt_len: Target length of input prompts in tokens
        output_len: Maximum number of tokens to generate
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8080")
        batch_size: Number of concurrent requests to send

    Returns:
        Dictionary containing throughput metrics and detailed request results
    """
    print(f"Starting throughput benchmark:")
    print(f"  Model: {model_id}")
    print(f"  Input prompt length: {input_prompt_len} tokens")
    print(f"  Output length: {output_len} tokens")
    print(f"  Batch size: {batch_size} concurrent requests")
    print(f"  Server: {base_url}")

    # Generate prompts for all requests
    prompts = [_generate_prompt(input_prompt_len) for _ in range(batch_size)]

    # Create connector for session to handle multiple concurrent requests
    connector = aiohttp.TCPConnector(
        limit=batch_size + 10,  # Add some buffer for connection pooling
        limit_per_host=batch_size + 5,
        keepalive_timeout=300,
        enable_cleanup_closed=True
    )

    overall_start_time = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for all concurrent requests
        tasks = [
            send_single_request(
                session=session,
                model_id=model_id,
                prompt=prompts[i],
                max_tokens=output_len,
                request_id=i,
                base_url=base_url
            )
            for i in range(batch_size)
        ]

        # Execute all requests concurrently
        print(f"Sending {batch_size} requests concurrently...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        overall_end_time = time.time()

    # Process results
    successful_requests = []
    failed_requests = []

    for result in results:
        if isinstance(result, Exception):
            failed_requests.append({
                "success": False,
                "error": str(result),
                "latency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "time_per_completion_token_seconds": 0,
                "time_per_total_token_seconds": 0,
                "time_to_first_token_seconds": 0,
                "time_between_tokens_seconds": [],
                "avg_time_between_tokens_seconds": 0,
                "prefill_time_seconds": 0,
                "decode_time_seconds": 0
            })
        elif result["success"]:
            successful_requests.append(result)
        else:
            failed_requests.append(result)

    # Calculate metrics
    total_time = overall_end_time - overall_start_time
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful_requests)
    total_tokens_processed = sum(r.get("total_tokens", 0) for r in successful_requests)

    # Latency statistics for successful requests
    latencies = [r["latency"] for r in successful_requests]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # Calculate throughput metrics
    requests_per_second = len(successful_requests) / total_time if total_time > 0 else 0
    completion_tokens_per_second = total_completion_tokens / total_time if total_time > 0 else 0
    input_tokens_per_second = sum(r.get("prompt_tokens", 0) for r in successful_requests) / total_time if total_time > 0 else 0

    # Calculate time per token statistics
    time_per_completion_token_values = [r["time_per_completion_token_seconds"] for r in successful_requests if r["time_per_completion_token_seconds"] > 0]
    time_per_total_token_values = [r["time_per_total_token_seconds"] for r in successful_requests if r["time_per_total_token_seconds"] > 0]

    # Average time per token metrics
    avg_time_per_completion_token = sum(time_per_completion_token_values) / len(time_per_completion_token_values) if time_per_completion_token_values else 0
    avg_time_per_total_token = sum(time_per_total_token_values) / len(time_per_total_token_values) if time_per_total_token_values else 0

    # Min/Max time per token metrics
    min_time_per_completion_token = min(time_per_completion_token_values) if time_per_completion_token_values else 0
    max_time_per_completion_token = max(time_per_completion_token_values) if time_per_completion_token_values else 0
    min_time_per_total_token = min(time_per_total_token_values) if time_per_total_token_values else 0
    max_time_per_total_token = max(time_per_total_token_values) if time_per_total_token_values else 0

    # Calculate streaming metrics
    prefill_times = [r["prefill_time_seconds"] for r in successful_requests if r["prefill_time_seconds"] > 0]
    decode_times = [r["decode_time_seconds"] for r in successful_requests if r["decode_time_seconds"] > 0]
    time_to_first_token_values = [r["time_to_first_token_seconds"] for r in successful_requests if r["time_to_first_token_seconds"] > 0]

    # Collect all time between tokens values
    all_time_between_tokens = []
    for r in successful_requests:
        all_time_between_tokens.extend(r.get("time_between_tokens_seconds", []))

    avg_time_between_tokens_values = [r["avg_time_between_tokens_seconds"] for r in successful_requests if r["avg_time_between_tokens_seconds"] > 0]

    # Calculate streaming statistics
    avg_prefill_time = sum(prefill_times) / len(prefill_times) if prefill_times else 0
    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
    avg_time_to_first_token = sum(time_to_first_token_values) / len(time_to_first_token_values) if time_to_first_token_values else 0
    avg_all_time_between_tokens = sum(all_time_between_tokens) / len(all_time_between_tokens) if all_time_between_tokens else 0
    avg_avg_time_between_tokens = sum(avg_time_between_tokens_values) / len(avg_time_between_tokens_values) if avg_time_between_tokens_values else 0

    print(f"\nBenchmark Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Successful requests: {len(successful_requests)}/{batch_size}")
    print(f"  Failed requests: {len(failed_requests)}")
    print(f"  Average latency: {avg_latency:.2f} seconds")
    print(f"  Min/Max latency: {min_latency:.2f}/{max_latency:.2f} seconds")
    print(f"  Requests per second: {requests_per_second:.2f}")
    print(f"  Completion tokens per second: {completion_tokens_per_second:.2f}")
    print(f"  Input tokens processed per second: {input_tokens_per_second:.2f}")
    print(f"  Total completion tokens: {total_completion_tokens}")

    # Time per token metrics
    if time_per_completion_token_values:
        print(f"  Time per completion token: {avg_time_per_completion_token*1000:.2f}ms avg, {min_time_per_completion_token*1000:.2f}/{max_time_per_completion_token*1000:.2f}ms min/max")
    if time_per_total_token_values:
        print(f"  Time per total token: {avg_time_per_total_token*1000:.2f}ms avg, {min_time_per_total_token*1000:.2f}/{max_time_per_total_token*1000:.2f}ms min/max")

    # Streaming metrics
    if prefill_times:
        print(f"  Time to first token (TTFT): {avg_time_to_first_token*1000:.2f}ms avg")
        print(f"  Prefill time: {avg_prefill_time*1000:.2f}ms avg")
    if decode_times:
        print(f"  Decode time: {avg_decode_time*1000:.2f}ms avg")
    if all_time_between_tokens:
        print(f"  Time between tokens: {avg_all_time_between_tokens*1000:.2f}ms avg")

    return {
        "config": {
            "model_id": model_id,
            "input_prompt_len": input_prompt_len,
            "output_len": output_len,
            "batch_size": batch_size,
            "base_url": base_url
        },
        "timing": {
            "total_time_seconds": total_time,
            "start_time": overall_start_time,
            "end_time": overall_end_time
        },
        "requests": {
            "total": batch_size,
            "successful": len(successful_requests),
            "failed": len(failed_requests),
            "success_rate": len(successful_requests) / batch_size if batch_size > 0 else 0
        },
        "throughput": {
            "requests_per_second": requests_per_second,
            "completion_tokens_per_second": completion_tokens_per_second,
            "input_tokens_per_second": input_tokens_per_second,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens_processed": total_tokens_processed
        },
        "latency": {
            "average_seconds": avg_latency,
            "minimum_seconds": min_latency,
            "maximum_seconds": max_latency,
            "all_latencies": latencies
        },
        "time_per_token": {
            "completion_tokens": {
                "average_seconds": avg_time_per_completion_token,
                "minimum_seconds": min_time_per_completion_token,
                "maximum_seconds": max_time_per_completion_token,
                "all_values": time_per_completion_token_values
            },
            "total_tokens": {
                "average_seconds": avg_time_per_total_token,
                "minimum_seconds": min_time_per_total_token,
                "maximum_seconds": max_time_per_total_token,
                "all_values": time_per_total_token_values
            }
        },
        "streaming_metrics": {
            "time_to_first_token": {
                "average_seconds": avg_time_to_first_token,
                "all_values": time_to_first_token_values
            },
            "prefill_time": {
                "average_seconds": avg_prefill_time,
                "all_values": prefill_times
            },
            "decode_time": {
                "average_seconds": avg_decode_time,
                "all_values": decode_times
            },
            "time_between_tokens": {
                "average_seconds": avg_all_time_between_tokens,
                "request_averages": {
                    "average_seconds": avg_avg_time_between_tokens,
                    "all_values": avg_time_between_tokens_values
                },
                "all_values": all_time_between_tokens
            }
        },
        "detailed_results": {
            "successful": successful_requests,
            "failed": failed_requests
        }
    }


def bench_throughput_sync(
        model_id: str,
        input_prompt_len: int,
        output_len: int,
        base_url: str,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
    """Synchronous wrapper for bench_throughput function.

    This is a convenience function that runs the async bench_throughput function
    in an asyncio event loop.

    Args:
        model_id: The model identifier to use for requests
        input_prompt_len: Target length of input prompts in tokens
        output_len: Maximum number of tokens to generate
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8080")
        batch_size: Number of concurrent requests to send

    Returns:
        Dictionary containing throughput metrics and detailed request results
    """
    return asyncio.run(bench_throughput(
        model_id=model_id,
        input_prompt_len=input_prompt_len,
        output_len=output_len,
        base_url=base_url,
        batch_size=batch_size
    ))
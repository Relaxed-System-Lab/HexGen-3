#!/usr/bin/env python3
"""
Benchmark script for AFD autoscaling metrics evaluation.

This script sends requests to SGLang AFD servers and collects client-side metrics
(TTFT and TBT), which will be aligned with server-side metrics collected by the
AFD metrics collector. After collecting metrics, it automatically analyzes and
visualizes the results.

Usage:
    python bench_afd_with_metrics.py \
        --api-url http://localhost:30000 \
        --num-requests 100 \
        --request-rate 2.0 \
        --server-metrics-dir /tmp/sglang_afd_metrics \
        --output-dir results
"""

import argparse
import asyncio
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from collections import defaultdict
from urllib.parse import urljoin

import aiohttp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    created_time: float
    first_token_time: Optional[float] = None  # Timestamp when first token arrived
    token_times: List[float] = None  # List of timestamps for each token arrival
    ttft: float = 0.0  # Time to first token
    tbt: float = 0.0   # Average time between tokens
    prompt_len: int = 0
    output_len: int = 0
    success: bool = False
    
    def __post_init__(self):
        if self.token_times is None:
            self.token_times = []
    
    def to_dict(self):
        return asdict(self)


def estimate_token_count(text: str) -> int:
    """Estimate token count from text (rough approximation: ~4 chars per token)."""
    return len(text) // 4


def ensure_min_tokens(prompt_template: str, min_tokens: int) -> str:
    """Ensure prompt has at least min_tokens by repeating if necessary."""
    current_tokens = estimate_token_count(prompt_template)
    if current_tokens >= min_tokens:
        return prompt_template
    
    # Repeat template until we have enough tokens
    multiplier = (min_tokens // current_tokens) + 1
    return (prompt_template + " ") * multiplier


def generate_varied_prompt(base_template: str, request_id: int, min_tokens: int, max_variation: float = 2.0) -> str:
    """Generate a prompt with varied length.
    
    Args:
        base_template: Base prompt template
        request_id: Request ID for uniqueness
        min_tokens: Minimum token count
        max_variation: Maximum variation multiplier (e.g., 2.0 means up to 2x the base length)
    
    Returns:
        A prompt with varied length
    """
    # Ensure base template meets minimum
    base_template = ensure_min_tokens(base_template, min_tokens)
    base_tokens = estimate_token_count(base_template)
    
    # Use request_id to create deterministic but varied lengths
    # Use a combination of sin and request_id to create variation
    variation_factor = 0.5 + 0.5 * math.sin(request_id * 0.5) + 0.3 * (request_id % 10) / 10.0
    # Normalize to [0, 1] range, then scale to [0.5, max_variation]
    variation_factor = 0.5 + (max_variation - 0.5) * variation_factor
    
    # Calculate target tokens with variation (always relative to base_tokens)
    target_tokens = int(base_tokens * variation_factor)
    target_tokens = max(min_tokens, target_tokens)  # Ensure minimum
    
    # Always generate varied length by repeating the template
    # Calculate how many times we need to repeat to reach target length
    if base_tokens == 0:
        base_tokens = 1  # Avoid division by zero
    
    multiplier = max(1, (target_tokens // base_tokens) + (1 if target_tokens % base_tokens > 0 else 0))
    varied_prompt = (base_template + " ") * multiplier
    
    # Add unique identifier
    varied_prompt = f"{varied_prompt} Request #{request_id}."
    
    return varied_prompt


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    max_tokens: int,
    request_id: int
) -> RequestMetrics:
    """Send a single request and collect metrics."""
    created_time = time.time()
    
    payload = {
        "model": "default",
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    metrics = RequestMetrics(
        created_time=created_time,
        first_token_time=None,
        token_times=[],
        ttft=0.0,
        tbt=0.0,
        prompt_len=estimate_token_count(prompt),
        output_len=0,
        success=False
    )
    
    try:
        async with session.post(
            urljoin(api_url, "/v1/completions"),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status == 200:
                first_token_time = None
                token_times = []
                
                # Streaming response: Server-Sent Events (SSE) format
                # Each chunk contains one or more newly generated tokens
                # Format: "data: {json}\n\n" for each token
                # The server sends tokens one by one as they are generated, NOT all at once
                async for line in response.content:
                    if line:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str == '[DONE]':
                                    break
                                
                                data = json.loads(data_str)
                                # Record time when this chunk arrives at client
                                # This is NOT the time to iterate, but the time when server sends this token
                                current_time = time.time()
                                
                                # Check if this chunk contains a new token
                                has_new_token = False
                                if 'choices' in data and len(data['choices']) > 0:
                                    # OpenAI Completions API format
                                    choice = data['choices'][0]
                                    if 'text' in choice and choice['text']:
                                        has_new_token = True
                                elif 'text' in data and data['text']:
                                    # Alternative format
                                    has_new_token = True
                                
                                if has_new_token:
                                    if metrics.first_token_time is None:
                                        # First token: calculate TTFT (Time To First Token)
                                        metrics.first_token_time = current_time
                                        metrics.ttft = metrics.first_token_time - created_time
                                    else:
                                        # Subsequent tokens: record time for TBT calculation
                                        metrics.token_times.append(current_time)
                                    
                                    # Count output tokens (each chunk typically contains 1 token)
                                    metrics.output_len += 1
                        
                        except (json.JSONDecodeError, KeyError):
                            pass
                
                metrics.success = True
                
                # Calculate average TBT (Time Between Tokens)
                # TBT is the average interval between consecutive tokens for this request
                if len(metrics.token_times) > 1:
                    intervals = [metrics.token_times[i] - metrics.token_times[i-1] 
                                for i in range(1, len(metrics.token_times))]
                    if intervals:
                        metrics.tbt = sum(intervals) / len(intervals)
                elif len(metrics.token_times) == 1 and metrics.first_token_time:
                    # Only one token after first token
                    metrics.tbt = metrics.token_times[0] - metrics.first_token_time
    
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        metrics.success = False
    
    return metrics


def load_sharegpt_prompts(dataset_path: str, num_requests: int, min_input_tokens: int = 256) -> List[str]:
    """Load prompts from ShareGPT dataset.
    
    Args:
        dataset_path: Path to ShareGPT JSON file. If file doesn't exist, will download it.
        num_requests: Number of prompts to sample
        min_input_tokens: Minimum input token count (used for filtering)
    
    Returns:
        List of prompt strings
    """
    import urllib.request
    
    SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    
    # Download if necessary
    if not os.path.isfile(dataset_path):
        print(f"ShareGPT dataset not found at {dataset_path}, downloading...")
        try:
            urllib.request.urlretrieve(SHAREGPT_URL, dataset_path)
            print(f"Downloaded ShareGPT dataset to {dataset_path}")
        except Exception as e:
            raise ValueError(f"Failed to download ShareGPT dataset: {e}")
    
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Filter out conversations with less than 2 turns
    dataset = [
        data for data in dataset 
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    
    # Extract prompts (first turn from each conversation)
    prompts = []
    for data in dataset:
        conversations = data.get("conversations", data.get("conversation", []))
        if len(conversations) >= 2 and conversations[0].get("from") == "human":
            prompt = conversations[0].get("value", "")
            # Rough token estimation: ~4 chars per token
            if estimate_token_count(prompt) >= min_input_tokens:
                prompts.append(prompt)
    
    # Shuffle and sample
    random.shuffle(prompts)
    
    if len(prompts) < num_requests:
        # Repeat if not enough
        prompts = (prompts * ((num_requests // len(prompts)) + 1))[:num_requests]
    else:
        # Sample if too many
        prompts = prompts[:num_requests]
    
    print(f"Loaded {len(prompts)} prompts from ShareGPT dataset")
    return prompts


def get_default_realistic_prompts() -> List[str]:
    """Get a list of realistic prompts for testing."""
    return [
        "Explain the difference between machine learning and deep learning in simple terms.",
        "What are the main causes of climate change and how can we address them?",
        "Describe the process of photosynthesis and why it's important for life on Earth.",
        "How does the human immune system work to protect us from diseases?",
        "What are the key principles of effective project management?",
        "Explain the concept of quantum computing and its potential applications.",
        "What is the difference between renewable and non-renewable energy sources?",
        "Describe the water cycle and its importance in Earth's ecosystem.",
        "How do neural networks learn and make predictions?",
        "What are the main challenges in developing sustainable transportation?",
        "Explain the theory of relativity in simple terms.",
        "What factors contribute to economic growth in developing countries?",
        "Describe the structure and function of DNA in living organisms.",
        "How does artificial intelligence impact modern healthcare?",
        "What are the key elements of effective communication?",
        "Explain the causes and effects of ocean acidification.",
        "What is the role of cryptography in modern computer security?",
        "Describe the process of cellular respiration.",
        "How do supply and demand determine market prices?",
        "What are the benefits and challenges of renewable energy adoption?",
        "Explain how vaccines work to prevent diseases.",
        "What are the main features of effective software design?",
        "Describe the formation and evolution of stars.",
        "How does blockchain technology ensure data integrity?",
        "What factors influence consumer purchasing decisions?",
        "Explain the greenhouse effect and its role in climate change.",
        "What are the key principles of sustainable agriculture?",
        "How do computers process and store information?",
        "Describe the relationship between stress and health.",
        "What are the main challenges in space exploration?",
    ]


async def send_requests(
    api_url: str,
    num_requests: int,
    request_rate: float,
    prompt_template: Optional[str] = None,
    prompt_file: Optional[str] = None,
    use_realistic_prompts: bool = False,
    sharegpt_path: Optional[str] = None,
    max_tokens: int = 256,
    min_input_tokens: int = 256,
    variable_load: bool = False
) -> List[RequestMetrics]:
    """Send multiple requests concurrently and collect metrics.
    
    Args:
        prompt_template: Template string for prompts (if provided, will be used with variation)
        prompt_file: Path to file containing prompts (one per line)
        use_realistic_prompts: If True, use a list of realistic prompts
        sharegpt_path: Path to ShareGPT dataset JSON file (if provided, will use ShareGPT prompts)
        variable_load: If True, use variable load pattern (bursts and quiet periods)
                      Example: 150 requests in 5s, wait 10s, then 50 requests in 10s
    """
    # Determine which prompts to use (priority: sharegpt > prompt_file > use_realistic_prompts > prompt_template > default)
    if sharegpt_path:
        prompts = load_sharegpt_prompts(sharegpt_path, num_requests, min_input_tokens)
        print(f"Using ShareGPT dataset from {sharegpt_path}")
    elif use_realistic_prompts:
        default_prompts = get_default_realistic_prompts()
        # Repeat or sample as needed
        if len(default_prompts) < num_requests:
            prompts = (default_prompts * ((num_requests // len(default_prompts)) + 1))[:num_requests]
        else:
            import random
            prompts = random.sample(default_prompts, num_requests)
        print(f"Using {len(prompts)} realistic prompts")
    elif prompt_template:
        # Use template with variation (existing behavior)
        prompts = None  # Will be generated per request
        print(f"Using template-based prompts with variation")
    else:
        # Default: use realistic prompts
        default_prompts = get_default_realistic_prompts()
        if len(default_prompts) < num_requests:
            prompts = (default_prompts * ((num_requests // len(default_prompts)) + 1))[:num_requests]
        else:
            import random
            prompts = random.sample(default_prompts, num_requests)
        print(f"Using {len(prompts)} realistic prompts (default)")
    
    # Create a connector with higher limit to allow more concurrent connections
    # Default limit is 100, which causes blocking when sending >100 requests simultaneously
    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Calculate send times for each request
        send_times = []
        start_time = time.time()
        cumulative_time = 0.0
        
        if variable_load:
            # Variable load pattern: bursts with simultaneous burst in the middle
            # Pattern: burst1 -> burst2 -> burst_wait (simultaneous) -> burst3 -> burst4 -> burst5
            burst1_count = 20
            burst1_duration = 10.0

            burst1_wait_duration = 0

            burst2_count = 40
            burst2_duration = 10.0

            burst2_wait_duration = 0

            burst3_count = 80
            burst3_duration = 10.0

            burst3_wait_duration = 0

            burst4_count = 40
            burst4_duration = 10.0

            burst4_wait_duration = 0

            burst5_count = 20
            burst5_duration = 10.0

            
            # First burst: distribute requests evenly over [0, burst1_duration]
            for i in range(burst1_count):
                if burst1_count > 1:
                    send_time_offset = (i / (burst1_count - 1)) * burst1_duration
                else:
                    send_time_offset = 0
                send_times.append(start_time + send_time_offset)
            
            # Second burst: distribute requests evenly over [burst1_duration, burst1_duration + burst2_duration]
            if burst2_count > 0:
                burst2_start_time = burst1_duration + burst1_wait_duration
                
                for i in range(burst2_count):
                    if burst2_count > 1:
                        send_time_offset = burst2_start_time + (i / (burst2_count - 1)) * burst2_duration
                    else:
                        send_time_offset = burst2_start_time
                    send_times.append(start_time + send_time_offset)

            # Third burst: distribute requests evenly over [burst_wait_start_time, burst_wait_start_time + burst3_duration]
            if burst3_count > 0:
                burst3_start_time = burst1_duration + burst1_wait_duration + burst2_duration + burst2_wait_duration
                
                for i in range(burst3_count):
                    if burst3_count > 1:
                        send_time_offset = burst3_start_time + (i / (burst3_count - 1)) * burst3_duration
                    else:
                        send_time_offset = burst3_start_time
                    send_times.append(start_time + send_time_offset)
        
            # Fourth burst: distribute requests evenly over [burst3_end, burst3_end + burst4_duration]
            if burst4_count > 0:
                burst4_start_time = burst1_duration + burst1_wait_duration + burst2_duration + burst2_wait_duration + burst3_duration + burst3_wait_duration
                
                for i in range(burst4_count):
                    if burst4_count > 1:
                        send_time_offset = burst4_start_time + (i / (burst4_count - 1)) * burst4_duration
                    else:
                        send_time_offset = burst4_start_time
                    send_times.append(start_time + send_time_offset)

            # Fifth burst: distribute requests evenly over [burst4_end, burst4_end + burst5_duration]
            if burst5_count > 0:
                burst5_start_time = burst1_duration + burst1_wait_duration + burst2_duration + burst2_wait_duration + burst3_duration + burst3_wait_duration + burst4_duration + burst4_wait_duration
                
                for i in range(burst5_count):
                    if burst5_count > 1:
                        send_time_offset = burst5_start_time + (i / (burst5_count - 1)) * burst5_duration
                    else:
                        send_time_offset = burst5_start_time
                    send_times.append(start_time + send_time_offset)
        else:
            # Constant rate
            for i in range(num_requests):
                interval = 1.0 / request_rate
                cumulative_time += interval
                send_times.append(start_time + cumulative_time)
        
        # Send requests concurrently at scheduled times
        async def send_at_time(i: int, send_time: float, prompt_list: Optional[List[str]], template: Optional[str]) -> RequestMetrics:
            """Send request at specified time."""
            wait_time = send_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Select or generate prompt
            if prompt_list is not None:
                # Use pre-loaded prompts
                prompt = prompt_list[i]
                # Ensure minimum length if needed
                if estimate_token_count(prompt) < min_input_tokens:
                    prompt = ensure_min_tokens(prompt, min_input_tokens)
            else:
                # Generate varied prompt from template
                prompt = generate_varied_prompt(template, i+1, min_input_tokens, max_variation=2.0)
            
            # Generate varied output length (256-512 tokens, with some variation)
            # Use request_id to create deterministic but varied lengths
            output_variation = 0.5 + 0.5 * math.sin((i+1) * 0.3) + 0.3 * ((i+1) % 7) / 7.0
            varied_max_tokens = int(max_tokens * (0.8 + 0.4 * output_variation))
            varied_max_tokens = max(256, varied_max_tokens)  # Ensure minimum 256
            
            return await send_request(session, api_url, prompt, varied_max_tokens, i+1)
        
        # Create tasks for all requests (concurrent)
        tasks = [
            asyncio.create_task(send_at_time(i, send_times[i], prompts, prompt_template))
            for i in range(num_requests)
        ]
        
        # Print progress periodically
        async def print_progress():
            completed = 0
            while completed < num_requests:
                await asyncio.sleep(1.0)
                completed = sum(1 for t in tasks if t.done())
                if completed < num_requests:
                    print(f"Progress: {completed}/{num_requests} requests completed")
        
        progress_task = asyncio.create_task(print_progress())
        
        # Wait for all requests to complete
        metrics_list = await asyncio.gather(*tasks)
        progress_task.cancel()
        
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
    
    return metrics_list


def aggregate_ttft_tbt_by_window(
    metrics_list: List[RequestMetrics],
    start_time: float,
    end_time: float,
    window_size: float = 1.0
) -> List[Dict]:
    """Aggregate TTFT and TBT metrics by time windows.
    
    Args:
        metrics_list: List of RequestMetrics objects
        start_time: Script start time (window array start)
        end_time: Current time when this function is called (window array end)
        window_size: Size of each time window in seconds (default: 1.0)
    
    Returns:
        List of dictionaries, each containing:
        - timestamp: Window start time
        - avg_ttft: Average TTFT for requests that received first token in this window
        - avg_tbt: Average TBT for tokens generated in this window
        - count_ttft: Number of TTFT samples in this window
        - count_tbt: Number of TBT samples (token intervals) in this window
    """
    if not metrics_list:
        return []
    
    # Filter successful metrics
    successful_metrics = [m for m in metrics_list if m.success and m.first_token_time]
    if not successful_metrics:
        return []
    
    # Create window array: from start_time to end_time, 1 second windows
    num_windows = int((end_time - start_time) / window_size) + 1
    windows = []
    for i in range(num_windows):
        window_start_time = start_time + i * window_size
        windows.append({
            "timestamp": int(window_start_time),
            "ttft_values": [],  # TTFT values in this window
            "tbt_values": []     # TBT values (token intervals) in this window
        })
    
    # Traverse all successful metrics once
    for m in successful_metrics:
        # 1. Check which window the first_token_time belongs to
        if m.first_token_time:
            window_idx = int((m.first_token_time - start_time) / window_size)
            if 0 <= window_idx < len(windows):
                windows[window_idx]["ttft_values"].append(m.ttft)
        
        # 2. Traverse token_times, group tokens by window and calculate TBT
        # Include first_token_time in the token list for interval calculation
        all_token_times = []
        if m.first_token_time:
            all_token_times.append(m.first_token_time)
        all_token_times.extend(m.token_times)
        
        if len(all_token_times) > 0:
            # Group tokens by window: {window_idx: [token_times]}
            window_token_times = defaultdict(list)
            for token_time in all_token_times:
                window_idx = int((token_time - start_time) / window_size)
                if 0 <= window_idx < len(windows):
                    window_token_times[window_idx].append(token_time)
            
            # For each window, calculate TBT from token intervals within that window
            for window_idx, token_times_in_window in window_token_times.items():
                # Sort token times in this window
                token_times_in_window.sort()
                # Calculate intervals between consecutive tokens in this window
                if len(token_times_in_window) > 1:
                    intervals = [
                        token_times_in_window[i] - token_times_in_window[i-1]
                        for i in range(1, len(token_times_in_window))
                    ]
                    windows[window_idx]["tbt_values"].extend(intervals)
    
    # Calculate averages for each window
    result = []
    for window in windows:
        avg_ttft = float(np.mean(window["ttft_values"])) if window["ttft_values"] else None
        avg_tbt = float(np.mean(window["tbt_values"])) if window["tbt_values"] else None
        
        result.append({
            "timestamp": window["timestamp"],
            "avg_ttft": avg_ttft,
            "avg_tbt": avg_tbt,
            "count_ttft": len(window["ttft_values"]),
            "count_tbt": len(window["tbt_values"])
        })
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark AFD with client-side metrics and analysis")
    parser.add_argument("--api-url", type=str, default="http://localhost:30000",
                       help="API URL of the SGLang server")
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests to send")
    parser.add_argument("--request-rate", type=float, default=2.0,
                       help="Request rate (requests per second)")
    parser.add_argument("--prompt-template", type=str, default=None,
                       help="Prompt template (will be used with variation to generate diverse prompts)")
    parser.add_argument("--prompt-file", type=str, default=None,
                       help="Path to file containing prompts (one prompt per line). If provided, will use these prompts instead of template.")
    parser.add_argument("--use-realistic-prompts", action="store_true",
                       help="Use a built-in list of realistic prompts for testing (default if no template or file specified)")
    parser.add_argument("--sharegpt-path", type=str, default=None,
                       help="Path to ShareGPT dataset JSON file. If provided, will use ShareGPT prompts. If file doesn't exist, will download it automatically.")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum tokens per request (minimum 256)")
    parser.add_argument("--min-input-tokens", type=int, default=256,
                       help="Minimum input tokens per request")
    parser.add_argument("--output", type=str, default="client_metrics.json",
                       help="Output JSON file for client metrics")
    parser.add_argument("--server-metrics-dir", type=str, default=None,
                       help="Directory containing server metrics JSONL files (if provided, analysis will be performed)")
    parser.add_argument("--output-dir", type=str, default="metric_results",
                       help="Output directory for analysis results")
    parser.add_argument("--plot-output", type=str, default="afd_metrics_analysis.png",
                       help="Output plot file name")
    parser.add_argument("--window-size", type=float, default=1.0,
                       help="Time window size for alignment (seconds)")
    parser.add_argument("--variable-load", action="store_true",
                       help="Use variable load pattern: 75%% requests in 5s burst, wait 10s, then remaining in 10s burst")
    
    args = parser.parse_args()
    
    print(f"Sending {args.num_requests} requests to {args.api_url}")
    print(f"Request rate: {args.request_rate} req/s ({'variable load' if args.variable_load else 'constant rate'})")
    
    print(f"Input tokens (min): {args.min_input_tokens}")
    print(f"Output tokens (max): {args.max_tokens}")
    
    # Send requests
    start_time = time.time()
    metrics_list = asyncio.run(send_requests(
        args.api_url,
        args.num_requests,
        args.request_rate,
        prompt_template=args.prompt_template,
        prompt_file=args.prompt_file,
        use_realistic_prompts=args.use_realistic_prompts or (args.prompt_template is None and args.prompt_file is None and args.sharegpt_path is None),
        sharegpt_path=args.sharegpt_path,
        max_tokens=args.max_tokens,
        min_input_tokens=args.min_input_tokens,
        variable_load=args.variable_load
    ))
    end_time = time.time()
    
    # Filter successful requests
    successful_metrics = [m for m in metrics_list if m.success]
    
    print(f"\nCompleted {len(successful_metrics)}/{len(metrics_list)} requests")
    print(f"Total time: {end_time - start_time:.2f}s")
    
    if successful_metrics:
        avg_ttft = sum(m.ttft for m in successful_metrics) / len(successful_metrics)
        avg_tbt = sum(m.tbt for m in successful_metrics) / len(successful_metrics)
        
        print(f"Average TTFT: {avg_ttft*1000:.2f}ms")
        print(f"Average TBT: {avg_tbt*1000:.2f}ms")
    
    # Aggregate TTFT and TBT by time windows (1 second windows)
    # Use script start_time and current end_time for window range
    current_time = time.time()
    time_series_metrics = aggregate_ttft_tbt_by_window(
        metrics_list, 
        start_time=start_time, 
        end_time=current_time,
        window_size=1.0
    )
    
    # Save to file
    output_data = {
        'time_series': time_series_metrics  # Add time-series aggregated metrics
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nClient metrics saved to {args.output}")
    print(f"Time-series metrics: {len(time_series_metrics)} windows")
    if time_series_metrics:
        windows_with_data = [w for w in time_series_metrics if w['count_ttft'] > 0 or w['count_tbt'] > 0]
        print(f"Windows with data: {len(windows_with_data)}")

if __name__ == "__main__":
    main()



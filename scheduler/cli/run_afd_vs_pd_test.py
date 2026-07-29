#!/usr/bin/env python3
"""
Compare baseline P-D disaggregated serving vs AFD serving on Alpaca traces.
"""

import os
import sys
import time
import json
import random
from typing import List, Dict, Tuple
from threading import Lock
import numpy as np

# Optional: Hugging Face dataset for Alpaca traces
HAS_DATASETS = False
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# Force synthetic traces for determinism (set to True to allow datasets)
USE_DATASETS = False

# Make simulator importable when run from CLI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.core.cluster_manager import ClusterManager, ClusterConfiguration, NodeConfiguration
from simulator.core.arrival import PoissonProcess

print_lock = Lock()

# Global seed for reproducibility
GLOBAL_SEED = 42


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def load_alpaca_traces(num_traces: int = 200) -> List[Dict]:
    """Load a small Alpaca-like trace list. Falls back to synthetic if datasets is unavailable."""
    traces: List[Dict] = []
    if HAS_DATASETS and USE_DATASETS:
        try:
            safe_print("Loading Alpaca dataset...")
            dataset = load_dataset("tatsu-lab/alpaca", split="train", num_proc=2, keep_in_memory=False)
            safe_print(f"Dataset loaded with {len(dataset)} examples")
            request_id = 0
            for example in dataset:
                if request_id >= num_traces:
                    break
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output_text = example.get("output", "")
                full_input = f"{instruction} {input_text}".strip()
                input_len = max(5, len(full_input) // 4)
                output_len = max(5, len(output_text) // 4)
                traces.append(
                    {
                        "request_id": f"req_{request_id}",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "input_length": input_len,
                        "output_length": output_len,
                    }
                )
                request_id += 1
        except Exception as e:
            safe_print(f"Warning: Failed to load Alpaca dataset, using synthetic traces: {e}")

    if not traces:
        safe_print("Using synthetic traces (datasets missing or failed).")
        import numpy as np
        rng = np.random.default_rng(GLOBAL_SEED)

        for i in range(num_traces):
            traces.append(
                {
                    "request_id": f"req_{i}",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input_length": int(rng.integers(128, 1024)),
                    "output_length": int(rng.integers(32, 512)),
                }
            )
    return traces


def generate_requests_with_arrivals(
    traces: List[Dict], num_requests: int, arrival_rate: float = 5.0
) -> Tuple[List[Dict], float]:
    """Generate a fixed list of requests and arrival times using a Poisson process."""
    rng = np.random.default_rng(GLOBAL_SEED)
    gaps = rng.exponential(1.0 / arrival_rate, num_requests)
    arrival_times = np.cumsum(gaps)

    requests = []
    for i in range(num_requests):
        trace = traces[i % len(traces)]
        requests.append(
            {
                "request_id": f"{trace['request_id']}_{i}",
                "model": trace["model"],
                "arrive_at": float(arrival_times[i]),
                "input_length": trace["input_length"],
                "output_length": trace["output_length"],
            }
        )
    total_duration = float(arrival_times.max()) + 10.0
    return requests, total_duration


class FixedTraceArrivalProcess:
    """Arrival process using pre-generated requests with fixed arrival times."""

    def __init__(self, requests: List[Dict]):
        self.requests = requests
        self._rate = len(requests) / (max(r["arrive_at"] for r in requests) + 1)
        self._cv = 1

    def rate(self):
        return self._rate

    def cv(self):
        return self._cv

    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        arrival_times = [r["arrive_at"] + start for r in self.requests]
        return [t for t in arrival_times if t < start + duration]

    def generate_workload(self, start: float, duration: float):
        workload = []
        for req in self.requests:
            if req["arrive_at"] + start < start + duration:
                r = req.copy()
                r["arrive_at"] = req["arrive_at"] + start
                workload.append(r)
        return workload

    def __str__(self):
        return f"FixedTraceArrivalProcess(requests={len(self.requests)}, rate={self.rate():.2f})"


def build_baseline_cluster(decode_hardware: str, decode_replicas: int, max_batch_size: int) -> ClusterConfiguration:
    """Baseline P-D separated cluster (no AFD)."""
    nodes: List[NodeConfiguration] = []
    # One prefill-only (reuse decode hardware for simplicity)
    nodes.append(
        NodeConfiguration(
            node_id=f"{decode_hardware}_prefill",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=decode_hardware,
            pd_separation=True,
            pd_prefill_only=True,
            kv_transfer_bandwidth_gbps=100.0,
        )
    )
    # Decode replicas
    for j in range(decode_replicas):
        nodes.append(
            NodeConfiguration(
                node_id=f"{decode_hardware}_decode_{j}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware=decode_hardware,
                max_batch_size=max_batch_size,
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0,
            )
        )
    return ClusterConfiguration(
        cluster_id=f"baseline_pd_{decode_hardware}",
        nodes=nodes,
        scheduler_algorithm="random",
    )


def build_afd_cluster(
    prefill_hardware: str,
    attention_hardware: str,
    ffn_hardware: str,
    attention_replicas: int,
    ffn_replicas: int,
    attention_batch_size: int,
    ffn_max_batch_size: int,
) -> ClusterConfiguration:
    """AFD cluster: prefill-only + attention nodes + FFN nodes."""
    nodes: List[NodeConfiguration] = []
    nodes.append(
        NodeConfiguration(
            node_id=f"{prefill_hardware}_prefill",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=prefill_hardware,
            pd_separation=True,
            pd_prefill_only=True,
            kv_transfer_bandwidth_gbps=100.0,
        )
    )
    for j in range(attention_replicas):
        nodes.append(
            NodeConfiguration(
                node_id=f"{attention_hardware}_attn_{j}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware=attention_hardware,
                max_batch_size=attention_batch_size,
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0,
                afd_attention=True,
            )
        )
    for j in range(ffn_replicas):
        nodes.append(
            NodeConfiguration(
                node_id=f"{ffn_hardware}_ffn_{j}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware=ffn_hardware,
                max_batch_size=ffn_max_batch_size,
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0,
                afd_ffn=True,
            )
        )
    return ClusterConfiguration(
        cluster_id=f"afd_{attention_hardware}_attn_{ffn_hardware}_ffn",
        nodes=nodes,
        scheduler_algorithm="random",
        afd_enabled=True,
        afd_attention_batch_size=attention_batch_size,
        afd_ffn_max_batch_size=ffn_max_batch_size,
        afd_activation_bandwidth_gbps=100.0,
    )


def run_scenario(
    name: str,
    cluster_config: ClusterConfiguration,
    arrival_process,
    duration: float,
) -> Dict:
    safe_print(f"\n=== Running scenario: {name} ===")
    start_time = time.time()
    cluster = ClusterManager(cluster_config, arrival_process)
    cluster.run_simulation(duration=duration, enable_failures=False)
    elapsed = time.time() - start_time
    results = cluster.get_results()
    summary = results.get("completed_requests", [])
    loop_stats = results.get("event_loop_stats", {}) if isinstance(results, dict) else {}
    # Use actual processing span: last completion - first arrival (fallback to loop time)
    if summary:
        first_arrival = min(r["arrive_at"] for r in summary)
        last_finish = max(r.get("generation_finished_at", 0) for r in summary)
        sim_time = max(last_finish - first_arrival, 0.0)
    else:
        sim_time = loop_stats.get("total_simulation_time", duration)

    latencies = [
        r["generation_finished_at"] - r["arrive_at"]
        for r in summary
        if r.get("generation_finished_at") is not None
    ]
    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx]

    stats = {
        "scenario": name,
        "completed": len(summary),
        "elapsed_wall": elapsed,
        "throughput": len(summary) / sim_time if sim_time > 0 else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency": pct(0.5),
        "p95_latency": pct(0.95),
        "p99_latency": pct(0.99),
        "sim_time": sim_time,
    }
    safe_print(
        f"{name}: completed={stats['completed']}, "
        f"avg/p50/p95/p99={stats['avg_latency']:.3f}/"
        f"{stats['p50_latency']:.3f}/"
        f"{stats['p95_latency']:.3f}/"
        f"{stats['p99_latency']:.3f} s, "
        f"throughput={stats['throughput']:.2f} req/s, "
        f"wall={stats['elapsed_wall']:.2f}s"
    )
    return {"stats": stats, "details": results}


def main():
    # Global seeds for reproducibility
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    traces = load_alpaca_traces(num_traces=200)
    requests, duration = generate_requests_with_arrivals(traces, num_requests=200, arrival_rate=5.0)
    arrival_process = FixedTraceArrivalProcess(requests)

    # Baseline: 4x H100 decode, max batch size 8
    baseline_cfg = build_baseline_cluster(
        decode_hardware="NVDA:H100:SXM",
        decode_replicas=4,
        max_batch_size=8,
    )

    # AFD: 3x H20 for Attention (batch=8), 1x H100 for FFN (max batch=24)
    afd_cfg = build_afd_cluster(
        prefill_hardware="NVDA:H100:SXM",
        attention_hardware="NVDA:H20",
        ffn_hardware="NVDA:H100:SXM",
        attention_replicas=3,
        ffn_replicas=1,
        attention_batch_size=8,
        ffn_max_batch_size=24,
    )

    baseline = run_scenario("baseline_pd", baseline_cfg, arrival_process, duration=duration)
    afd = run_scenario("afd", afd_cfg, arrival_process, duration=duration)

    safe_print("\n=== Summary ===")
    safe_print(json.dumps({"baseline_pd": baseline["stats"], "afd": afd["stats"]}, indent=2))


if __name__ == "__main__":
    main()

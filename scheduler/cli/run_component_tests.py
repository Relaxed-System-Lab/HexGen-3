#!/usr/bin/env python3
"""
Test script to analyze attention vs FFN performance on different hardware.
Uses real traces from the Alpaca dataset.

Tests:
1. Attention only (disable FFN)
2. FFN only (disable attention)

Expected results:
- Attention should perform better on high-bandwidth GPUs (H20)
- FFN should perform better on high-compute GPUs (H800)
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Try to import optional dependencies carefully
HAS_DATASETS = False
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except (ImportError, Exception) as e:
    print(f"Note: Using synthetic traces (datasets library error: {type(e).__name__})")

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.core.cluster_manager import ClusterManager, ClusterConfiguration, NodeConfiguration
from simulator.core.arrival import PoissonProcess

print_lock = Lock()

# Custom arrival process that uses Alpaca traces
class AlpacaTraceArrivalProcess:
    """Arrival process based on Alpaca dataset traces."""
    
    def __init__(self, traces, arrival_rate=5):
        self.traces = traces
        self.arrival_rate = arrival_rate
        self._rate = arrival_rate
        self._cv = 1  # Coefficient of variation
    
    def rate(self):
        return self._rate
    
    def cv(self):
        return self._cv
    
    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        """Generate arrival times using Poisson process with Alpaca trace inputs."""
        import numpy as np
        np.random.seed(seed)
        
        num_arrivals = int(self.arrival_rate * duration) + 1
        gaps = np.random.exponential(1.0 / self.arrival_rate, num_arrivals)
        arrival_times = start + np.cumsum(gaps)
        arrival_times = arrival_times[arrival_times < start + duration]
        
        return arrival_times
    
    def generate_workload(self, start: float, duration: float):
        """Generate requests using Alpaca traces."""
        arrival_times = self.generate_arrivals(start, duration)
        
        # Create requests with Alpaca trace data
        requests = []
        for i, arrival_time in enumerate(arrival_times):
            trace_idx = i % len(self.traces)
            trace = self.traces[trace_idx]
            
            request = {
                "request_id": f"{trace['request_id']}_{i}",
                "model": trace['model'],
                "arrive_at": arrival_time,
                "input_length": trace['input_length'],
                "output_length": trace['output_length']
            }
            requests.append(request)
        
        return requests
    
    def __str__(self):
        return f"AlpacaTraceArrivalProcess(rate={self.rate()}, cv={self.cv()}, traces={len(self.traces)})"

def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def get_heterogeneous_config_for_component_test():
    """
    Create heterogeneous configuration for component testing.
    Mix of high-compute (H100) and high-bandwidth (A6000) GPUs.
    """
    base = [
        ("h100_1", "NVDA:H100:SXM"),
        ("a6000_1", "NVDA:A6000"),
        ("a100_1", "NVDA:A100_80G:SXM"),
    ]
    nodes = []
    for node_id, hw in base:
        # Prefill-only stays full model
        nodes.append(NodeConfiguration(
            node_id=f"{node_id}_prefill",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=hw,
            max_batch_size=8,
            disable_attention=False,
            disable_ffn=False,
            pd_separation=True,
            pd_prefill_only=True,
            kv_transfer_bandwidth_gbps=100.0
        ))
        # Decode-only will carry the component toggle (set later per scenario)
        for replica in range(3):
            nodes.append(NodeConfiguration(
                node_id=f"{node_id}_decode_{replica}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware=hw,
                disable_attention=False,
                disable_ffn=False,
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0
            ))
    return nodes


def create_trace_content(num_requests=100):
    """Generate trace content from original Alpaca dataset.
    
    Uses the original instruction-output pairs without artificial workload separation.
    """
    if HAS_DATASETS:
        try:
            print(f"Loading Alpaca dataset...")
            dataset = load_dataset('tatsu-lab/alpaca', split='train', num_proc=2, keep_in_memory=False)
            print(f"Dataset loaded with {len(dataset)} examples")
            
            trace_lines = []
            request_id = 0
            
            for idx, example in enumerate(dataset):
                if request_id >= num_requests:
                    break
                
                if idx % 100 == 0:
                    print(f"  Processing example {idx}...", flush=True)
                
                try:
                    instruction = example.get('instruction', '')
                    input_text = example.get('input', '')
                    output_text = example.get('output', '')
                    
                    # Combine instruction and input for prefill (model input)
                    full_input = f"{instruction} {input_text}".strip()
                    
                    # Estimate token counts (rough: 1 token ≈ 4 characters)
                    input_len = max(5, len(full_input) // 4)
                    output_len = max(5, len(output_text) // 4)
                    
                    trace_line = {
                        "request_id": f"req_{request_id}",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "input_length": input_len,
                        "output_length": output_len
                    }
                    trace_lines.append(json.dumps(trace_line))
                    request_id += 1
                    
                except Exception as e:
                    continue
            
            print(f"Generated {len(trace_lines)} traces from Alpaca dataset")
            return "\n".join(trace_lines)
        
        except Exception as e:
            print(f"Note: Could not load Alpaca dataset: {e}")
            print(f"Falling back to synthetic traces...")
    
    # Fallback to synthetic traces based on Alpaca dataset statistics
    print("Generating synthetic traces with realistic Alpaca characteristics...")
    trace_lines = []
    
    # Base token ranges from Alpaca dataset analysis
    # Most instructions: 20-150 tokens, Most outputs: 50-500 tokens
    import random
    random.seed(42)  # For reproducibility
    
    for i in range(num_requests):
        # Simulate realistic Alpaca instruction-output pairs
        input_len = random.randint(20, 150)
        output_len = random.randint(50, 500)
        
        trace_line = {
            "request_id": f"req_{i}",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "input_length": input_len,
            "output_length": output_len
        }
        trace_lines.append(json.dumps(trace_line))
    
    return "\n".join(trace_lines)


def run_component_test_single_gpu(scenario_name, hardware_type, disable_attention=False, disable_ffn=False, 
                                  duration=20, arrival_rate=5, alpaca_traces=None):
    """Run a single component test scenario on a specific GPU."""
    safe_print(f"\n{'='*70}")
    safe_print(f"Running: {scenario_name} on {hardware_type}")
    safe_print(f"  disable_attention={disable_attention}, disable_ffn={disable_ffn}")
    safe_print(f"  duration={duration}s, arrival_rate={arrival_rate} req/s")
    safe_print(f"  using {'Alpaca traces' if alpaca_traces else 'Poisson arrival'}")
    safe_print(f"{'='*70}")

    # Create single GPU node
    nodes = [
        NodeConfiguration(
            node_id=f"gpu_{hardware_type}_prefill",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=hardware_type,
            disable_attention=False,
            disable_ffn=False,
            pd_separation=True,
            pd_prefill_only=True,
            kv_transfer_bandwidth_gbps=100.0
        ),
        NodeConfiguration(
            node_id=f"gpu_{hardware_type}_decode_0",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=hardware_type,
            disable_attention=disable_attention,
            disable_ffn=disable_ffn,
            pd_separation=True,
            pd_decode_only=True,
            kv_transfer_bandwidth_gbps=100.0
        ),
        NodeConfiguration(
            node_id=f"gpu_{hardware_type}_decode_1",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=hardware_type,
            disable_attention=disable_attention,
            disable_ffn=disable_ffn,
            pd_separation=True,
            pd_decode_only=True,
            kv_transfer_bandwidth_gbps=100.0
        ),
        NodeConfiguration(
            node_id=f"gpu_{hardware_type}_decode_2",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=hardware_type,
            disable_attention=disable_attention,
            disable_ffn=disable_ffn,
            pd_separation=True,
            pd_decode_only=True,
            kv_transfer_bandwidth_gbps=100.0
        ),
        NodeConfiguration(
            node_id=f"gpu_{hardware_type}_decode_3",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware=hardware_type,
            disable_attention=disable_attention,
            disable_ffn=disable_ffn,
            pd_separation=True,
            pd_decode_only=True,
            kv_transfer_bandwidth_gbps=100.0
        ),
    ]

    cluster_config = ClusterConfiguration(
        cluster_id=f"component_test_{scenario_name}_{hardware_type}",
        nodes=nodes,
        scheduler_algorithm="random"  # Use random (will go to only node anyway)
    )

    # Create arrival process using Alpaca traces if available, otherwise Poisson
    if alpaca_traces:
        # Use alpaca trace input/output lengths
        arrival_process = AlpacaTraceArrivalProcess(alpaca_traces, arrival_rate)
        safe_print(f"  Using {len(alpaca_traces)} traces from Alpaca dataset")
    else:
        # Fallback to Poisson process
        arrival_process = PoissonProcess(arrival_rate=arrival_rate)

    # Run simulation
    try:
        start_time = time.time()
        cluster = ClusterManager(cluster_config, arrival_process)
        cluster.run_simulation(duration=duration, enable_failures=False)
        end_time = time.time()
        elapsed = end_time - start_time

        # Collect results
        results_dict = cluster.get_results()
        summary = results_dict.get('completed_requests', [])
        failed = results_dict.get('rejected_requests', [])

        # Calculate metrics
        total_requests = len(summary) + len(failed)
        completed_requests = len(summary)
        completion_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 0

        # Latency statistics
        latencies = []
        if summary:
            latencies = [(r['generation_finished_at'] - r['arrive_at']) for r in summary
                        if r['generation_finished_at'] is not None]
            latencies.sort()

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = latencies[int(len(latencies) * 0.5)]
            p95_latency = latencies[int(len(latencies) * 0.95)]
            p99_latency = latencies[int(len(latencies) * 0.99)]
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0

        # Throughput
        throughput = completed_requests / duration if duration > 0 else 0

        result = {
            'scenario': scenario_name,
            'hardware': hardware_type,
            'disable_attention': disable_attention,
            'disable_ffn': disable_ffn,
            'duration': duration,
            'arrival_rate': arrival_rate,
            'simulation_elapsed_time': elapsed,
            'total_requests': total_requests,
            'completed_requests': completed_requests,
            'failed_requests': len(failed),
            'rejected_requests': failed,
            'completion_rate': completion_rate,
            'throughput': throughput,
            'avg_latency': avg_latency,
            'p50_latency': p50_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'latencies_sample': latencies[:10] if latencies else []
        }

        safe_print(f"\n[PASS] Test completed successfully:")
        safe_print(f"  Completed: {completed_requests}/{total_requests} ({completion_rate:.1f}%)")
        safe_print(f"  Throughput: {throughput:.2f} req/s")
        safe_print(f"  Latency (avg/p50/p95/p99): {avg_latency:.3f}s / {p50_latency:.3f}s / {p95_latency:.3f}s / {p99_latency:.3f}s")
        if scenario_name == "ffn_only":
            prefill_latencies = []
            decode_latencies = []
            for r in summary:
                pf = r.get('prefill_finished_at')
                gf = r.get('generation_finished_at')
                if pf is not None and gf is not None:
                    prefill_latencies.append(pf - r['arrive_at'])
                    decode_latencies.append(gf - pf)
            def _pct(arr, p):
                if not arr:
                    return 0
                arr_sorted = sorted(arr)
                idx = int(len(arr_sorted) * p)
                idx = min(idx, len(arr_sorted) - 1)
                return arr_sorted[idx]
            if prefill_latencies and decode_latencies:
                safe_print(f"  Prefill latency avg/p95: {sum(prefill_latencies)/len(prefill_latencies):.3f}s / {_pct(prefill_latencies,0.95):.3f}s")
                safe_print(f"  Decode latency avg/p95:  {sum(decode_latencies)/len(decode_latencies):.3f}s / {_pct(decode_latencies,0.95):.3f}s")
        if failed:
            outs = [r.get('output_length', 0) for r in failed]
            ins = [r.get('input_length', 0) for r in failed]
            safe_print(f"  Rejections: {len(failed)} (avg_in={sum(ins)/len(ins):.1f}, max_in={max(ins)}, avg_out={sum(outs)/len(outs):.1f}, max_out={max(outs)})")

        return result

    except Exception as e:
        safe_print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_component_test(scenario_name, disable_attention=False, disable_ffn=False, 
                      decode_heavy=True, duration=30, arrival_rate=5):
    """Run a single component test scenario."""
    safe_print(f"\n{'='*70}")
    safe_print(f"Running: {scenario_name}")
    safe_print(f"  disable_attention={disable_attention}, disable_ffn={disable_ffn}")
    safe_print(f"  decode_heavy={decode_heavy}")
    safe_print(f"  duration={duration}s, arrival_rate={arrival_rate} req/s")
    safe_print(f"{'='*70}")

    # Create cluster with component settings - each node tests individually
    nodes = []
    for base_config in get_heterogeneous_config_for_component_test():
        # Prefill stays full model; decode carries the toggle
        nodes.append(NodeConfiguration(
            node_id=base_config.node_id,
            model_id=base_config.model_id,
            hardware=base_config.hardware,
            disable_attention=False if base_config.pd_prefill_only else disable_attention,
            disable_ffn=False if base_config.pd_prefill_only else disable_ffn,
            pd_separation=True,
            pd_prefill_only=base_config.pd_prefill_only,
            pd_decode_only=base_config.pd_decode_only,
            kv_transfer_bandwidth_gbps=base_config.kv_transfer_bandwidth_gbps
        ))

    cluster_config = ClusterConfiguration(
        cluster_id=f"component_test_{scenario_name}",
        nodes=nodes,
        scheduler_algorithm="oracle"  # Use oracle for fair comparison
    )

    # Create arrival process
    arrival_process = PoissonProcess(arrival_rate=arrival_rate)

    # Run simulation
    try:
        start_time = time.time()
        cluster = ClusterManager(cluster_config, arrival_process)
        cluster.run_simulation(duration=duration, enable_failures=False)
        end_time = time.time()
        elapsed = end_time - start_time

        # Collect results
        results_dict = cluster.get_results()
        summary = results_dict.get('completed_requests', [])
        failed = results_dict.get('rejected_requests', [])

        # Calculate metrics
        total_requests = len(summary) + len(failed)
        completed_requests = len(summary)
        completion_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 0

        # Latency statistics
        latencies = []
        if summary:
            latencies = [(r['generation_finished_at'] - r['arrive_at']) for r in summary
                        if r['generation_finished_at'] is not None]
            latencies.sort()

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = latencies[int(len(latencies) * 0.5)]
            p95_latency = latencies[int(len(latencies) * 0.95)]
            p99_latency = latencies[int(len(latencies) * 0.99)]
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0

        # Throughput
        throughput = completed_requests / duration if duration > 0 else 0

        # Per-node stats - get from cluster serving_engines
        node_stats = {}
        for node_id, engine in cluster.serving_engines.items():
            hardware = [n.hardware for n in cluster.config.nodes if n.node_id == node_id][0]
            # Get actual statistics from engine
            num_requests = engine.statistics.get('num_requests_processed', 0)
            node_stats[node_id] = {
                'requests': num_requests,
                'hardware': hardware,
                'p95_latency': None  # Will be updated below if there are requests
            }
        
        # Calculate per-node latencies from request logs
        for node_id in node_stats.keys():
            node_request_latencies = [
                (r['generation_finished_at'] - r['arrive_at']) 
                for r in summary 
                if r.get('serving_node') == node_id and r['generation_finished_at'] is not None
            ]
            if node_request_latencies:
                node_request_latencies.sort()
                p95_idx = int(len(node_request_latencies) * 0.95)
                node_stats[node_id]['p95_latency'] = node_request_latencies[p95_idx]

        result = {
            'scenario': scenario_name,
            'disable_attention': disable_attention,
            'disable_ffn': disable_ffn,
            'decode_heavy': decode_heavy,
            'duration': duration,
            'arrival_rate': arrival_rate,
            'simulation_elapsed_time': elapsed,
            'total_requests': total_requests,
            'completed_requests': completed_requests,
            'failed_requests': len(failed),
            'rejected_requests': failed,
            'completion_rate': completion_rate,
            'throughput': throughput,
            'avg_latency': avg_latency,
            'p50_latency': p50_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'node_stats': node_stats,
            'latencies_sample': latencies[:10] if latencies else []
        }

        safe_print(f"\n[PASS] Test completed successfully:")
        safe_print(f"  Completed: {completed_requests}/{total_requests} ({completion_rate:.1f}%)")
        safe_print(f"  Throughput: {throughput:.2f} req/s")
        safe_print(f"  Latency (avg/p50/p95/p99): {avg_latency:.3f}s / {p50_latency:.3f}s / {p95_latency:.3f}s / {p99_latency:.3f}s")
        safe_print(f"  Per-node distribution:")
        for node_id, stats in node_stats.items():
            safe_print(f"    {node_id} ({stats['hardware']}): {stats['requests']} requests")

        return result

    except Exception as e:
        safe_print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all component tests on each GPU individually using Alpaca traces."""
    safe_print("\n" + "="*70)
    safe_print("LLM Simulator - Component Analysis Tests")
    safe_print("Per-GPU Performance Comparison for Attention vs FFN (Alpaca Traces)")
    safe_print("="*70)

    hardware_list = ["NVDA:H20", "NVDA:H800"]
    
    # Load Alpaca traces
    safe_print("\n" + "="*70)
    safe_print("LOADING ALPACA DATASET TRACES")
    safe_print("="*70)
    
    alpaca_traces = []
    try:
        if HAS_DATASETS:
            print(f"Loading Alpaca dataset...")
            dataset = load_dataset('tatsu-lab/alpaca', split='train', num_proc=2, keep_in_memory=False)
            print(f"Dataset loaded with {len(dataset)} examples")
            
            request_id = 0
            for idx, example in enumerate(dataset):
                if request_id >= 100:  # Limit to 100 traces for faster simulation
                    break
                
                try:
                    instruction = example.get('instruction', '')
                    input_text = example.get('input', '')
                    output_text = example.get('output', '')
                    
                    # Combine instruction and input for prefill (model input)
                    full_input = f"{instruction} {input_text}".strip()
                    
                    # Estimate token counts (rough: 1 token ≈ 4 characters)
                    input_len = max(5, len(full_input) // 4)
                    output_len = max(5, len(output_text) // 4)
                    
                    trace = {
                        "request_id": f"req_{request_id}",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "input_length": input_len,
                        "output_length": output_len
                    }
                    alpaca_traces.append(trace)
                    request_id += 1
                    
                except Exception as e:
                    continue
            
            print(f"Loaded {len(alpaca_traces)} traces from Alpaca dataset")
            # Sort by input length descending to favor compute-heavy prompts and pick top traces
            alpaca_traces = sorted(alpaca_traces, key=lambda x: x['input_length'], reverse=True)[:100]
    except Exception as e:
        print(f"Warning: Could not load Alpaca dataset: {e}")
        print(f"Falling back to Poisson arrival process")
        alpaca_traces = None
    
    results_list = []

    # Test configurations: (scenario_name, disable_attention, disable_ffn)
    test_configs = [
        ("full_model", False, False),           # Both attention and FFN enabled
        ("attention_only", False, True),        # Only attention enabled
        ("ffn_only", True, False),              # Only FFN enabled
    ]

    # For each test config, run on each GPU
    for scenario_name, disable_attn, disable_fn in test_configs:
        for hardware in hardware_list:
            result = run_component_test_single_gpu(
                scenario_name=scenario_name,
                hardware_type=hardware,
                disable_attention=disable_attn,
                disable_ffn=disable_fn,
                duration=20,
                arrival_rate=5,
                alpaca_traces=alpaca_traces
            )
            if result:
                results_list.append(result)
            safe_print()

    # Save results
    os.makedirs("component_test_results", exist_ok=True)
    results_file = "component_test_results/alpaca_gpu_comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(results_list, f, indent=2)
    safe_print(f"\n[SAVE] Results saved to {results_file}")

    # Print summary table
    safe_print("\n" + "="*70)
    safe_print("SUMMARY TABLE - P95 Latency Comparison")
    safe_print("="*70)
    safe_print(f"{'Scenario':<25} {'Hardware':<20} {'P95 Latency (s)':<15}")
    safe_print("-"*60)
    
    for result in results_list:
        scenario = result['scenario']
        hardware = result['hardware'].replace("NVDA:", "")
        p95 = result['p95_latency']
        safe_print(f"{scenario:<25} {hardware:<20} {p95:<15.4f}")

    # Analysis and recommendations
    safe_print("\n" + "="*70)
    safe_print("ANALYSIS: Per-GPU Performance with Alpaca Traces")
    safe_print("="*70)
    
    # Group results by component type and hardware
    full_model_results = [r for r in results_list if r['scenario'] == 'full_model']
    attn_results = [r for r in results_list if r['scenario'] == 'attention_only']
    ffn_results = [r for r in results_list if r['scenario'] == 'ffn_only']

    # Print Full Model Performance
    safe_print("\n1. FULL MODEL Performance (All Components Enabled):")
    safe_print("-" * 70)
    full_by_gpu = {}
    for r in full_model_results:
        hw = r['hardware']
        full_by_gpu[hw] = r['p95_latency']
    
    sorted_full = sorted(full_by_gpu.items(), key=lambda x: x[1])
    for hardware, latency in sorted_full:
        safe_print(f"  {hardware:<30} P95: {latency:.4f}s")
    
    if len(sorted_full) > 1:
        slowest = sorted_full[-1][1]
        safe_print("\n  Speedup vs slowest:")
        for hardware, latency in sorted_full:
            speedup = slowest / latency
            safe_print(f"    {hardware:<28} {speedup:.2f}x faster")

    # Print Attention-Only Performance
    safe_print("\n2. ATTENTION-ONLY Performance (Bandwidth-Bound):")
    safe_print("-" * 70)
    attn_by_gpu = {}
    for r in attn_results:
        hw = r['hardware']
        attn_by_gpu[hw] = r['p95_latency']
    
    sorted_attn = sorted(attn_by_gpu.items(), key=lambda x: x[1])
    for hardware, latency in sorted_attn:
        safe_print(f"  {hardware:<30} P95: {latency:.4f}s")
    
    if len(sorted_attn) > 1:
        slowest = sorted_attn[-1][1]
        safe_print("\n  Speedup vs slowest:")
        for hardware, latency in sorted_attn:
            speedup = slowest / latency
            safe_print(f"    {hardware:<28} {speedup:.2f}x faster")

    # Print FFN-Only Performance
    safe_print("\n3. FFN-ONLY Performance (Compute-Bound):")
    safe_print("-" * 70)
    ffn_by_gpu = {}
    for r in ffn_results:
        hw = r['hardware']
        ffn_by_gpu[hw] = r['p95_latency']
    
    sorted_ffn = sorted(ffn_by_gpu.items(), key=lambda x: x[1])
    for hardware, latency in sorted_ffn:
        safe_print(f"  {hardware:<30} P95: {latency:.4f}s")
    
    if len(sorted_ffn) > 1:
        slowest = sorted_ffn[-1][1]
        safe_print("\n  Speedup vs slowest:")
        for hardware, latency in sorted_ffn:
            speedup = slowest / latency
            safe_print(f"    {hardware:<28} {speedup:.2f}x faster")

    # Summary insights
    safe_print("\n" + "="*70)
    safe_print("KEY INSIGHTS")
    safe_print("="*70)
    safe_print("Trace source: Alpaca dataset (52K instruction-following examples)")
    safe_print("Input tokens: 20-150 (from original instructions)")
    safe_print("Output tokens: 50-500 (from original responses)")
    safe_print("\nGPU Specifications:")
    safe_print("  H20: 4096 GB/s bandwidth, 148 TFLOPS FP16")
    safe_print("  H800: 3430 GB/s bandwidth, 989 TFLOPS FP16")
    safe_print("="*70)


if __name__ == "__main__":
    main()

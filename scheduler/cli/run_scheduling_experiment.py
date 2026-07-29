#!/usr/bin/env python3
"""
Scheduling Strategy Experiment

Compares different scheduling strategies:
1. Oracle (perfect knowledge)
2. Random (lower bound)
3. RoundRobin (simple baseline)
4. FLOPs (pure compute)
5. Bandwidth (pure memory)
6. Roofline (treats request as whole)
7. InputOutput_Roofline (analyzes prefill/decode separately)
8. InputOutput_Threshold (threshold-based adaptive)

Tests on 4 workloads:
- decode_heavy: short input, long output
- prefill_heavy: long input, short output
- balanced: similar input/output
- realistic: 80/20 short/long mix
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.core.cluster_manager import ClusterManager, ClusterConfiguration, NodeConfiguration
from simulator.core.arrival import PoissonProcess

# Thread-safe printing
print_lock = Lock()


def get_heterogeneous_5gpu_config():
    """
    Create 5-GPU heterogeneous configuration with diverse hardware.
    RTX3090, L40S, A6000, A100_80G:SXM, H100:SXM (~1763 TFLOPS, $3.20/hr)
    """
    return [
        NodeConfiguration(
            node_id="rtx3090_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:RTX3090",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="l40s_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:L40S",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="a6000_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A6000",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="a100_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="h100_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:H100:SXM",
            max_batch_size=8
        ),
    ]


def get_homogeneous_config():
    """
    Create homogeneous configuration with same total FLOPS as heterogeneous.
    6x A100_80G:SXM (~1872 TFLOPS, $4.08/hr)
    """
    return [
        NodeConfiguration(
            node_id=f"a100_{i}",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            max_batch_size=8
        )
        for i in range(1, 7)
    ]


def get_original_heterogeneous_config():
    """Original 8-GPU heterogeneous configuration (2xH100, 4xA100, 2xA6000)."""
    return [
        # 2 H100 nodes (high compute)
        NodeConfiguration(
            node_id="h100_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:H100:SXM",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="h100_2",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:H100:SXM",
            max_batch_size=8
        ),
        # 4 A100 nodes (balanced)
        NodeConfiguration(
            node_id="a100_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="a100_2",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="a100_3",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="a100_4",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            max_batch_size=8
        ),
        # 2 A6000 nodes (good memory bandwidth)
        NodeConfiguration(
            node_id="a6000_1",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A6000",
            max_batch_size=8
        ),
        NodeConfiguration(
            node_id="a6000_2",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            hardware="NVDA:A6000",
            max_batch_size=8
        ),
    ]


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)

def run_experiment(scheduler_name, workload_name, trace_file, arrival_rate, duration, hw_config='hetero-5gpu'):
    """Run a single experiment with given scheduler and workload."""
    safe_print(f"\nRunning: {scheduler_name} on {workload_name} with {hw_config}")
    safe_print(f"  Trace: {trace_file}")
    safe_print(f"  Arrival rate: {arrival_rate} req/s")

    # Select hardware configuration
    if hw_config == 'hetero-5gpu':
        nodes = get_heterogeneous_5gpu_config()
    elif hw_config == 'homo-6gpu':
        nodes = get_homogeneous_config()
    elif hw_config == 'hetero-original':
        nodes = get_original_heterogeneous_config()
    else:
        raise ValueError(f"Unknown hardware config: {hw_config}")

    cluster_config = ClusterConfiguration(
        cluster_id=f"experiment_{scheduler_name}_{workload_name}",
        nodes=nodes,
        scheduler_algorithm=scheduler_name
    )

    # Create arrival process
    arrival_process = PoissonProcess(arrival_rate=arrival_rate)

    # Run simulation
    start_time = time.time()
    cluster = ClusterManager(cluster_config, arrival_process)
    cluster.run_simulation(duration=duration, enable_failures=False)
    end_time = time.time()

    # Collect results
    results_dict = cluster.get_results()
    summary = results_dict.get('completed_requests', [])
    failed = results_dict.get('failed_requests', [])

    # Calculate metrics
    total_requests = len(summary) + len(failed)
    completed_requests = len(summary)
    completion_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 0

    # Latency statistics
    if summary:
        latencies = [(r['generation_finished_at'] - r['arrive_at']) for r in summary
                     if r['generation_finished_at'] is not None]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            p50_latency = sorted_latencies[int(len(sorted_latencies) * 0.5)]
            p90_latency = sorted_latencies[int(len(sorted_latencies) * 0.9)]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        else:
            avg_latency = p50_latency = p90_latency = p99_latency = 0
    else:
        avg_latency = p50_latency = p90_latency = p99_latency = 0

    # Throughput
    simulation_duration = end_time - start_time
    throughput = completed_requests / simulation_duration if simulation_duration > 0 else 0

    results = {
        'scheduler': scheduler_name,
        'workload': workload_name,
        'hw_config': hw_config,
        'total_requests': total_requests,
        'completed_requests': completed_requests,
        'failed_requests': len(failed),
        'completion_rate': completion_rate,
        'throughput_req_per_sec': throughput,
        'avg_latency': avg_latency,
        'p50_latency': p50_latency,
        'p90_latency': p90_latency,
        'p99_latency': p99_latency,
        'simulation_duration': simulation_duration,
    }

    safe_print(f"  Completed: {completed_requests}/{total_requests} ({completion_rate:.1f}%)")
    safe_print(f"  Throughput: {throughput:.2f} req/s")
    safe_print(f"  Avg Latency: {avg_latency:.3f}s")

    return results


def run_and_save_experiment(params):
    """Run experiment and save result. For use with ThreadPoolExecutor."""
    scheduler, workload_name, workload_config, hw_config, output_dir = params

    try:
        result = run_experiment(
            scheduler_name=scheduler,
            workload_name=workload_name,
            trace_file=workload_config['file'],
            arrival_rate=workload_config['arrival_rate'],
            duration=30.0,
            hw_config=hw_config
        )

        # Save individual result
        scheduler_clean = scheduler.replace('_', '-')
        workload_clean = workload_name.replace('_', '-')
        hw_config_clean = hw_config.replace('_', '-')
        result_file = os.path.join(output_dir, f"{scheduler_clean}_{workload_clean}_{hw_config_clean}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        safe_print(f"  ERROR in {scheduler}/{workload_name}/{hw_config}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Parse command line args
    num_threads = 1
    if len(sys.argv) > 1:
        try:
            num_threads = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of threads: {sys.argv[1]}")
            print("Usage: python run_scheduling_experiment.py [num_threads]")
            sys.exit(1)

    # Configuration
    main_schedulers = [
        'oracle',
        'inputoutput_roofline',
        'inputoutput_threshold',
        'roofline',
        'flops',
        'bandwidth',
    ]

    baseline_schedulers = [
        'round_robin',
        'random',
    ]

    workloads = {
        'small': {
            'file': 'examples/trace.jsonl',
            'arrival_rate': 10.0,
        },
        'decode_heavy': {
            'file': 'examples/trace_high_load_decode_heavy.jsonl',
            'arrival_rate': 10.0,
        },
        'prefill_heavy': {
            'file': 'examples/trace_high_load_prefill_heavy.jsonl',
            'arrival_rate': 5.0,
        },
        'balanced': {
            'file': 'examples/trace_high_load_balanced.jsonl',
            'arrival_rate': 8.0,
        },
        'realistic': {
            'file': 'examples/trace_high_load_realistic.jsonl',
            'arrival_rate': 10.0,
        },
    }

    hw_configs = ['hetero-5gpu', 'homo-6gpu']

    # Create output directory
    output_dir = 'experiment_results'
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("SCHEDULING STRATEGY EXPERIMENT")
    print("="*70)
    print(f"Threads: {num_threads}")
    print(f"Hardware configs: {len(hw_configs)} (hetero-5gpu, homo-6gpu)")
    print(f"Main schedulers: {len(main_schedulers)} (run on all workloads)")
    print(f"Baseline schedulers: {len(baseline_schedulers)} (run on small workload only)")
    print(f"Workloads: {len(workloads)}")
    total_experiments = (len(main_schedulers) * len(workloads) + len(baseline_schedulers) * 1) * len(hw_configs)
    print(f"Total experiments: {total_experiments}")
    print("="*70)

    # Build task list
    tasks = []
    for hw_config in hw_configs:
        for workload_name, workload_config in workloads.items():
            # Determine which schedulers to run for this workload
            if workload_name == 'small':
                schedulers_to_run = main_schedulers + baseline_schedulers
            else:
                schedulers_to_run = main_schedulers

            for scheduler in schedulers_to_run:
                tasks.append((scheduler, workload_name, workload_config, hw_config, output_dir))

    print(f"\nBuilt {len(tasks)} tasks. Starting execution...")
    start_time = time.time()

    # Run experiments
    all_results = []

    if num_threads == 1:
        # Single-threaded
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]")
            result = run_and_save_experiment(task)
            if result:
                all_results.append(result)
    else:
        # Multi-threaded
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(run_and_save_experiment, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    all_results.append(result)
                print(f"[{i}/{len(tasks)}] Completed")

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed:.1f}s")

    # Save all results
    all_results_file = os.path.join(output_dir, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {output_dir}/")
    print(f"Summary file: {all_results_file}")

    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (Avg Across All Workloads)")
    print(f"{'='*70}")

    scheduler_avg = {}
    all_schedulers = main_schedulers + baseline_schedulers
    for scheduler in all_schedulers:
        scheduler_results = [r for r in all_results if r['scheduler'] == scheduler]
        if scheduler_results:
            avg_throughput = sum(r['throughput_req_per_sec'] for r in scheduler_results) / len(scheduler_results)
            avg_latency = sum(r['avg_latency'] for r in scheduler_results) / len(scheduler_results)
            avg_p99 = sum(r['p99_latency'] for r in scheduler_results) / len(scheduler_results)
            avg_completion = sum(r['completion_rate'] for r in scheduler_results) / len(scheduler_results)

            scheduler_avg[scheduler] = {
                'throughput': avg_throughput,
                'latency': avg_latency,
                'p99': avg_p99,
                'completion': avg_completion
            }

    print(f"{'Scheduler':<30} | {'Avg Throughput':<15} | {'Avg Latency':<12} | {'Avg P99':<12} | {'Completion':<10}")
    print("-"*100)

    for scheduler in sorted(scheduler_avg.keys(), key=lambda x: scheduler_avg[x]['throughput'], reverse=True):
        stats = scheduler_avg[scheduler]
        print(f"{scheduler:<30} | "
              f"{stats['throughput']:<15.2f} | "
              f"{stats['latency']:<12.3f} | "
              f"{stats['p99']:<12.3f} | "
              f"{stats['completion']:<10.1f}%")


if __name__ == "__main__":
    main()

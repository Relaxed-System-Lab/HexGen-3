#!/usr/bin/env python3
"""Run the HexGen-3 style hierarchical scheduling framework.

Example:
  python3 cli/run_hexgen3_scheduler.py \
    --capacity '{"NVDA:H100:SXM": 4, "NVDA:H20": 4}' \
    --arrival-rate 4 --long-ratio 0.3 --iterations 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.scheduling import (  # noqa: E402
    GlobalSchedulerConfig,
    HexGenSchedulingFramework,
    LocalSchedulerConfig,
    WorkloadProfile,
    plan_to_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="HexGen-3 scheduling framework evaluator")
    parser.add_argument(
        "--capacity",
        default='{"NVDA:H100:SXM": 4, "NVDA:H20": 4}',
        help="JSON map of hardware name to GPU count",
    )
    parser.add_argument("--arrival-rate", type=float, default=4.0)
    parser.add_argument("--long-ratio", type=float, default=0.25)
    parser.add_argument("--short-input", type=int, default=512)
    parser.add_argument("--long-input", type=int, default=4096)
    parser.add_argument("--short-output", type=int, default=128)
    parser.add_argument("--long-output", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--stability-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=0, help="0 chooses the paper-style model-size heuristic")
    parser.add_argument("--model-size-billions", type=float, default=8.0)
    parser.add_argument("--uniform-only", action="store_true", help="Disable non-uniform TP partition enumeration")
    parser.add_argument("--enable-ep", action="store_true", help="Enable expert-parallel candidates")
    parser.add_argument("--num-experts", type=int, default=1)
    parser.add_argument("--cost-aware", action="store_true", help="Apply warm-start reload-cost filtering")
    parser.add_argument("--stability-window-s", type=float, default=300.0)
    parser.add_argument("--reload-bandwidth-gbps", type=float, default=600.0)
    parser.add_argument("--model-size-gb", type=float, default=16.0)
    parser.add_argument("--routing-policy", default="random")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    capacity = json.loads(args.capacity)
    workload = WorkloadProfile.synthetic(
        arrival_rate=args.arrival_rate,
        short_input=args.short_input,
        long_input=args.long_input,
        short_output=args.short_output,
        long_output=args.long_output,
        long_ratio=args.long_ratio,
        max_batch_size=args.max_batch_size,
    )
    framework = HexGenSchedulingFramework(
        local_config=LocalSchedulerConfig(
            enumerate_non_uniform=not args.uniform_only,
            enable_expert_parallel=args.enable_ep,
            num_experts=args.num_experts,
            cost_aware=args.cost_aware,
            stability_window_s=args.stability_window_s,
            reload_bandwidth_gbps=args.reload_bandwidth_gbps,
            model_size_gb=args.model_size_gb,
        ),
        global_config=GlobalSchedulerConfig(
            iterations=args.iterations,
            stability_iterations=args.stability_iterations,
            seed=args.seed,
            block_size=args.block_size,
            model_size_billions=args.model_size_billions,
        ),
        routing_policy=args.routing_policy,
    )
    plan = framework.optimize(workload, capacity)
    result = plan_to_dict(plan)
    result["workload"] = {
        "arrival_rate": workload.arrival_rate,
        "mean_input": workload.mean_input,
        "mean_output": workload.mean_output,
        "mean_decode_context": workload.mean_decode_context,
        "max_batch_size": workload.max_batch_size,
    }

    print("HexGen-3 scheduling result")
    print(f"  system throughput: {result['system_throughput_req_s']:.4f} req/s")
    print(f"  estimated latency: {result['estimated_latency_s']:.4f} s")
    print(
        "  tail latency: "
        f"p50={result['tail_latency_s'].get('p50', 0.0):.4f}s, "
        f"p95={result['tail_latency_s'].get('p95', 0.0):.4f}s, "
        f"p99={result['tail_latency_s'].get('p99', 0.0):.4f}s"
    )
    print(f"  cost: ${result['cost_per_hour']:.4f}/hr")
    print(f"  req per dollar: {result['req_per_dollar']:.4f}")
    print("  worker throughput:")
    for worker, value in result["throughput_req_s"].items():
        print(f"    {worker}: {value:.4f} req/s")
    print("  allocation:")
    for worker, hw_map in result["allocation"].items():
        print(f"    {worker}: {hw_map}")
    print("  parallelism:")
    for key, value in result["parallelism"].items():
        print(f"    {key}: {value}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

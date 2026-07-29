#!/usr/bin/env python3
"""Mimic the HexGen-3 paper experiments with the local scheduling framework.

This runner is intentionally explicit about approximations:
- WildGPT is represented by the compact WildChat-derived trace in
  ``traces/wildgpt_trace.jsonl``.
- Qwen3-30B-A3B uses the local ModelAnalyzer's Qwen3 fallback config when the
  Hugging Face config is unavailable.
- Baselines are approximated with the same roofline estimator so results are
  comparable within this simulator, not exact paper numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.scheduling import (  # noqa: E402
    AllocationMatrix,
    GlobalSchedulerConfig,
    HexGenSchedulingFramework,
    LocalScheduler,
    LocalSchedulerConfig,
    ParallelismStrategy,
    SimulatorEstimator,
    ThroughputProfile,
    WorkloadProfile,
    plan_to_dict,
)


H100 = "NVDA:H100:SXM"
H20 = "NVDA:H20"
PAPER_HOMO_CAPACITY = {H100: 16}
PAPER_HETERO_CAPACITY = {H100: 8, H20: 16}
PAPER_HOMO_COST = 49.12
PAPER_HETERO_COST = 54.16
PAPER_UNIT_PRICES = {H100: PAPER_HOMO_COST / 16, H20: (PAPER_HETERO_COST - PAPER_HOMO_COST / 2) / 16}
DYNAMIC_REQUEST_LOADS = [6600, 11200, 61400, 129000, 13900, 8200]
DYNAMIC_WORKLOAD_TYPES = [3, 3, 1, 1, 4, 2]
DYNAMIC_RESOURCE_BUDGETS = [12, 18, 18, 24, 14, 12]


@dataclass
class ExperimentMetrics:
    baseline: str
    architecture: str
    capacity: Dict[str, int]
    cost_per_hour: float
    system_throughput_req_s: float
    per_dollar_throughput_req_per_dollar: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    p99_latency_cost_dollar: float
    allocation: Dict[str, Dict[str, int]]
    parallelism: Dict[str, object]
    notes: str


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-style HexGen-3 experiments")
    parser.add_argument("--trace", default="traces/wildgpt_trace.jsonl")
    parser.add_argument("--output", default="results/hexgen3_paper_mimic/results.json")
    parser.add_argument("--summary-md", default="results/hexgen3_paper_mimic/summary.md")
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--arrival-rate", type=float, default=4.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--stability-iterations", type=int, default=8)
    parser.add_argument("--max-local-strategies", type=int, default=1024)
    parser.add_argument("--disable-ep", action="store_true", help="Disable HexGen-3 EP candidates")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--skip-dynamic", action="store_true")
    return parser.parse_args()


def load_trace(path: Path, samples: int) -> List[Dict[str, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"trace is empty: {path}")
    if samples <= 0 or samples >= len(rows):
        return rows
    step = len(rows) / samples
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(samples)]


def workload_from_trace(
    rows: List[Dict[str, int]],
    arrival_rate: float,
    max_batch_size: int,
    type_id: int = 1,
) -> WorkloadProfile:
    inputs = []
    outputs = []
    input_scale, output_scale = workload_type_scale(type_id)
    for row in rows:
        inputs.append(max(1, int(round(row["input_tokens"] * input_scale))))
        outputs.append(max(1, int(round(row["output_tokens"] * output_scale))))
    return WorkloadProfile(
        arrival_rate=arrival_rate,
        input_lengths=tuple(inputs),
        output_lengths=tuple(outputs),
        max_batch_size=max_batch_size,
    )


def workload_type_scale(type_id: int) -> Tuple[float, float]:
    # Paper Table 1 maps types 1-4 to WildGPT, OpenThoughts, OpenR1, NuminaMath.
    # Only WildGPT is materialized locally; the others are mimicked by scaling
    # the WildGPT token distribution toward reasoning/math-heavy workloads.
    return {
        1: (1.0, 1.0),
        2: (1.2, 2.4),
        3: (2.0, 3.0),
        4: (1.5, 2.8),
    }.get(type_id, (1.0, 1.0))


def base_local_config(
    args,
    uniform_only: bool,
    cost_aware: bool = False,
    enable_ep: Optional[bool] = None,
) -> LocalSchedulerConfig:
    if enable_ep is None:
        enable_ep = not args.disable_ep
    return LocalSchedulerConfig(
        node_gpus=8,
        enumerate_non_uniform=not uniform_only,
        max_local_strategies=args.max_local_strategies,
        enable_expert_parallel=enable_ep,
        num_experts=args.num_experts,
        cost_aware=cost_aware,
        stability_window_s=3600.0,
        model_size_gb=60.0,
    )


def evaluate_afd(
    name: str,
    workload: WorkloadProfile,
    capacity: Mapping[str, int],
    cost_per_hour: float,
    args,
    optimized_allocation: bool,
    uniform_parallelism: bool,
    enable_ep: bool,
    notes: str,
) -> ExperimentMetrics:
    framework = HexGenSchedulingFramework(
        model_id=args.model_id,
        local_config=base_local_config(args, uniform_parallelism, enable_ep=enable_ep),
        global_config=GlobalSchedulerConfig(
            iterations=args.iterations,
            stability_iterations=args.stability_iterations,
            block_size=1,
            model_size_billions=30,
            seed=7,
        ),
    )
    if optimized_allocation:
        plan = framework.optimize(workload, capacity)
    else:
        plan = framework.evaluate_allocation(workload, AllocationMatrix.uniform(capacity))
    return metrics_from_plan(name, "AFD", plan, capacity, cost_per_hour, notes)


def evaluate_pd(
    name: str,
    workload: WorkloadProfile,
    capacity: Mapping[str, int],
    cost_per_hour: float,
    args,
    optimized_split: bool,
    uniform_parallelism: bool,
    enable_ep: bool,
    notes: str,
) -> ExperimentMetrics:
    estimator = SimulatorEstimator(model_id=args.model_id)
    local = LocalScheduler(
        estimator,
        base_local_config(args, uniform_parallelism, enable_ep=enable_ep),
    )
    if optimized_split:
        plan = best_pd_split(local, estimator, workload, capacity)
    else:
        plan = fixed_pd_split(local, estimator, workload, capacity)
    plan_dict = pd_plan_to_dict(plan)
    return metrics_from_pd_dict(name, plan_dict, capacity, cost_per_hour, notes)


def fixed_pd_split(local, estimator, workload, capacity):
    pre_alloc = {hw: max(1, count // 3) for hw, count in capacity.items() if count > 1}
    decode_alloc = {hw: count - pre_alloc.get(hw, 0) for hw, count in capacity.items()}
    return evaluate_pd_split(local, estimator, workload, pre_alloc, decode_alloc)


def best_pd_split(local, estimator, workload, capacity):
    hardware = list(capacity)
    best = None
    for pre_counts in iter_pre_splits(hardware, capacity):
        decode_counts = {hw: capacity[hw] - pre_counts.get(hw, 0) for hw in hardware}
        if sum(pre_counts.values()) <= 0 or sum(decode_counts.values()) <= 0:
            continue
        plan = evaluate_pd_split(local, estimator, workload, pre_counts, decode_counts)
        if best is None or plan["system_throughput_req_s"] > best["system_throughput_req_s"]:
            best = plan
    if best is None:
        raise RuntimeError("could not find a valid PD split")
    return best


def iter_pre_splits(hardware: List[str], capacity: Mapping[str, int]):
    if not hardware:
        yield {}
        return
    head = hardware[0]
    tail = hardware[1:]
    for count in range(0, capacity[head] + 1):
        for rest in iter_pre_splits(tail, capacity):
            result = dict(rest)
            result[head] = count
            yield result


def evaluate_pd_split(local, estimator, workload, pre_alloc, decode_alloc):
    pre_throughput = 0.0
    decode_throughput = 0.0
    pre_strategies = {}
    decode_strategies = {}
    for hardware, gpus in pre_alloc.items():
        if gpus <= 0:
            continue
        slice_plan = local.choose_strategy("pre", hardware, gpus, workload)
        pre_throughput += slice_plan.throughput
        if slice_plan.strategy is not None:
            pre_strategies[hardware] = slice_plan.strategy

    for hardware, gpus in decode_alloc.items():
        if gpus <= 0:
            continue
        strategy, throughput = choose_decode_strategy(local, estimator, hardware, gpus, workload)
        decode_throughput += throughput
        if strategy is not None:
            decode_strategies[hardware] = strategy

    throughput = ThroughputProfile({"pre": pre_throughput, "attn": decode_throughput, "ffn": decode_throughput})
    latency = estimator.estimate_latency(workload, throughput)
    tails = estimator.estimate_tail_latency(workload, throughput)
    return {
        "allocation": {
            "pre": dict(pre_alloc),
            "decode": dict(decode_alloc),
        },
        "parallelism": {
            "pre": {hw: strategy.as_dict() for hw, strategy in pre_strategies.items()},
            "decode": {hw: strategy.as_dict() for hw, strategy in decode_strategies.items()},
        },
        "throughput_req_s": {
            "pre": pre_throughput,
            "decode": decode_throughput,
        },
        "system_throughput_req_s": throughput.bottleneck,
        "estimated_latency_s": latency,
        "tail_latency_s": tails,
    }


def choose_decode_strategy(local, estimator, hardware, gpus, workload):
    best_strategy = None
    best_throughput = 0.0
    for strategy in local.enumerate_strategies(gpus):
        throughput = estimate_decode_throughput(estimator, hardware, strategy, workload)
        if throughput > best_throughput:
            best_throughput = throughput
            best_strategy = strategy
    return best_strategy, best_throughput


def estimate_decode_throughput(estimator, hardware, strategy: ParallelismStrategy, workload):
    attn = estimator.estimate_slice_throughput("attn", hardware, strategy, workload)
    ffn = estimator.estimate_slice_throughput("ffn", hardware, strategy, workload)
    if attn <= 0 or ffn <= 0:
        return 0.0
    return 1.0 / ((1.0 / attn) + (1.0 / ffn))


def metrics_from_plan(
    name: str,
    architecture: str,
    plan,
    capacity: Mapping[str, int],
    cost_per_hour: float,
    notes: str,
) -> ExperimentMetrics:
    plan_dict = plan_to_dict(plan)
    return metrics_from_common(
        name=name,
        architecture=architecture,
        capacity=capacity,
        cost_per_hour=cost_per_hour,
        system_throughput=plan_dict["system_throughput_req_s"],
        latency=plan_dict["estimated_latency_s"],
        tails=plan_dict["tail_latency_s"],
        allocation=plan_dict["allocation"],
        parallelism=plan_dict["parallelism"],
        notes=notes,
    )


def metrics_from_pd_dict(
    name: str,
    plan_dict,
    capacity: Mapping[str, int],
    cost_per_hour: float,
    notes: str,
) -> ExperimentMetrics:
    return metrics_from_common(
        name=name,
        architecture="PD",
        capacity=capacity,
        cost_per_hour=cost_per_hour,
        system_throughput=plan_dict["system_throughput_req_s"],
        latency=plan_dict["estimated_latency_s"],
        tails=plan_dict["tail_latency_s"],
        allocation=plan_dict["allocation"],
        parallelism=plan_dict["parallelism"],
        notes=notes,
    )


def metrics_from_common(
    name,
    architecture,
    capacity,
    cost_per_hour,
    system_throughput,
    latency,
    tails,
    allocation,
    parallelism,
    notes,
) -> ExperimentMetrics:
    p99 = float(tails.get("p99", math.inf))
    return ExperimentMetrics(
        baseline=name,
        architecture=architecture,
        capacity=dict(capacity),
        cost_per_hour=cost_per_hour,
        system_throughput_req_s=system_throughput,
        per_dollar_throughput_req_per_dollar=system_throughput * 3600.0 / cost_per_hour,
        mean_latency_s=latency,
        p50_latency_s=float(tails.get("p50", math.inf)),
        p95_latency_s=float(tails.get("p95", math.inf)),
        p99_latency_s=p99,
        p99_latency_cost_dollar=p99 * cost_per_hour / 3600.0,
        allocation=allocation,
        parallelism=parallelism,
        notes=notes,
    )


def pd_plan_to_dict(plan):
    return plan


def hetero_capacity_for_total(total_gpus: int) -> Dict[str, int]:
    h100 = min(8, max(1, round(total_gpus / 3)))
    h20 = min(16, max(1, total_gpus - h100))
    while h100 + h20 > total_gpus and h20 > 1:
        h20 -= 1
    while h100 + h20 > total_gpus and h100 > 1:
        h100 -= 1
    return {H100: h100, H20: h20}


def cost_for_capacity(capacity: Mapping[str, int]) -> float:
    return sum(PAPER_UNIT_PRICES[hw] * count for hw, count in capacity.items())


def run_static(rows, args):
    workload = workload_from_trace(rows, args.arrival_rate, args.max_batch_size, type_id=1)
    return [
        evaluate_pd(
            "SGLang_Homo_PD",
            workload,
            PAPER_HOMO_CAPACITY,
            PAPER_HOMO_COST,
            args,
            optimized_split=False,
            uniform_parallelism=True,
            enable_ep=False,
            notes="Homogeneous PD baseline with fixed 1/3 prefill and 2/3 decode split.",
        ),
        evaluate_afd(
            "MegaScaleInfer_Homo_AFD",
            workload,
            PAPER_HOMO_CAPACITY,
            PAPER_HOMO_COST,
            args,
            optimized_allocation=False,
            uniform_parallelism=True,
            enable_ep=False,
            notes="Homogeneous AFD baseline with uniform allocation and no HexGen-3 local EP/non-uniform search.",
        ),
        evaluate_pd(
            "HexGen2_Hetero_PD",
            workload,
            PAPER_HETERO_CAPACITY,
            PAPER_HETERO_COST,
            args,
            optimized_split=True,
            uniform_parallelism=False,
            enable_ep=False,
            notes="Heterogeneous PD baseline with optimized prefill/decode split but no AFD-specific EP search.",
        ),
        evaluate_afd(
            "HexGen3_Homo_AFD",
            workload,
            PAPER_HOMO_CAPACITY,
            PAPER_HOMO_COST,
            args,
            optimized_allocation=True,
            uniform_parallelism=False,
            enable_ep=not args.disable_ep,
            notes="Full scheduler on homogeneous H100 AFD.",
        ),
        evaluate_afd(
            "HexGen3_Hetero_AFD",
            workload,
            PAPER_HETERO_CAPACITY,
            PAPER_HETERO_COST,
            args,
            optimized_allocation=True,
            uniform_parallelism=False,
            enable_ep=not args.disable_ep,
            notes="Full scheduler on heterogeneous H100/H20 AFD.",
        ),
        evaluate_afd(
            "HexGen3_Homo_UniformAllocation",
            workload,
            PAPER_HOMO_CAPACITY,
            PAPER_HOMO_COST,
            args,
            optimized_allocation=False,
            uniform_parallelism=False,
            enable_ep=not args.disable_ep,
            notes="Ablation: local parallelism optimized, global allocation fixed uniform.",
        ),
        evaluate_afd(
            "HexGen3_Homo_UniformParallelism",
            workload,
            PAPER_HOMO_CAPACITY,
            PAPER_HOMO_COST,
            args,
            optimized_allocation=True,
            uniform_parallelism=True,
            enable_ep=not args.disable_ep,
            notes="Ablation: global allocation optimized, local search restricted to uniform replicas.",
        ),
    ]


def run_dynamic(rows, args):
    dynamic = []
    for hour, (load, type_id, resources) in enumerate(
        zip(DYNAMIC_REQUEST_LOADS, DYNAMIC_WORKLOAD_TYPES, DYNAMIC_RESOURCE_BUDGETS),
        start=1,
    ):
        arrival_rate = load / 3600.0
        workload = workload_from_trace(rows, arrival_rate, args.max_batch_size, type_id=type_id)
        hetero_capacity = hetero_capacity_for_total(resources)
        hetero_cost = cost_for_capacity(hetero_capacity)
        homo_capacity = {H100: min(16, resources)}
        homo_cost = PAPER_UNIT_PRICES[H100] * homo_capacity[H100]
        dynamic.append(
            {
                "hour": hour,
                "request_load": load,
                "workload_type": type_id,
                "resource_budget_gpus": resources,
                "baselines": [
                    asdict(
                        evaluate_pd(
                            "SGLang_Autoscale_PD",
                            workload,
                            homo_capacity,
                            homo_cost,
                            args,
                            optimized_split=False,
                            uniform_parallelism=True,
                            enable_ep=False,
                            notes="Dynamic homogeneous PD baseline with paper Table-1 load.",
                        )
                    ),
                    asdict(
                        evaluate_pd(
                            "HeteroScale_Autoscale_PD",
                            workload,
                            hetero_capacity,
                            hetero_cost,
                            args,
                            optimized_split=True,
                            uniform_parallelism=False,
                            enable_ep=False,
                            notes="Dynamic heterogeneous PD autoscaling approximation.",
                        )
                    ),
                    asdict(
                        evaluate_afd(
                            "HexGen3_Autoscale_AFD",
                            workload,
                            hetero_capacity,
                            hetero_cost,
                            args,
                            optimized_allocation=True,
                            uniform_parallelism=False,
                            enable_ep=not args.disable_ep,
                            notes="Dynamic heterogeneous AFD scheduling approximation.",
                        )
                    ),
                ],
            }
        )
    return dynamic


def write_summary(path: Path, payload):
    lines = [
        "# HexGen-3 Paper-Mimic Experiment Results",
        "",
        "These are simulator estimates, not a reproduction of the paper's private run.",
        "",
        "## Static WildGPT",
        "",
        "| Baseline | Arch | Capacity | Throughput req/s | Req/$ | P99 s | P99 cost $ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["static_results"]:
        lines.append(
            f"| {item['baseline']} | {item['architecture']} | {item['capacity']} | "
            f"{item['system_throughput_req_s']:.4f} | "
            f"{item['per_dollar_throughput_req_per_dollar']:.2f} | "
            f"{item['p99_latency_s']:.4f} | {item['p99_latency_cost_dollar']:.6f} |"
        )

    if payload.get("dynamic_results"):
        lines.extend(["", "## Dynamic Table-1 Mimic", ""])
        for hour in payload["dynamic_results"]:
            lines.append(
                f"Hour {hour['hour']}: load={hour['request_load']}, "
                f"type={hour['workload_type']}, resources={hour['resource_budget_gpus']} GPUs"
            )
            lines.append("")
            lines.append("| Baseline | Throughput req/s | Req/$ | P99 s | Capacity |")
            lines.append("| --- | ---: | ---: | ---: | --- |")
            for item in hour["baselines"]:
                lines.append(
                    f"| {item['baseline']} | {item['system_throughput_req_s']:.4f} | "
                    f"{item['per_dollar_throughput_req_per_dollar']:.2f} | "
                    f"{item['p99_latency_s']:.4f} | {item['capacity']} |"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = load_trace(Path(args.trace), args.samples)
    static_results = run_static(rows, args)
    payload = {
        "experiment": "HexGen-3 paper mimic",
        "trace": args.trace,
        "trace_samples": len(rows),
        "model_id": args.model_id,
        "paper_configs": {
            "homogeneous": {"capacity": PAPER_HOMO_CAPACITY, "cost_per_hour": PAPER_HOMO_COST},
            "heterogeneous": {"capacity": PAPER_HETERO_CAPACITY, "cost_per_hour": PAPER_HETERO_COST},
            "dynamic_request_loads": DYNAMIC_REQUEST_LOADS,
            "dynamic_workload_types": DYNAMIC_WORKLOAD_TYPES,
            "dynamic_resource_budgets": DYNAMIC_RESOURCE_BUDGETS,
        },
        "metric_definitions": {
            "per_dollar_throughput_req_per_dollar": "system_throughput_req_s * 3600 / cost_per_hour",
            "p99_latency_cost_dollar": "p99_latency_s * cost_per_hour / 3600",
        },
        "static_results": [asdict(result) for result in static_results],
    }
    if not args.skip_dynamic:
        payload["dynamic_results"] = run_dynamic(rows, args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(Path(args.summary_md), payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.summary_md}")
    for result in static_results:
        print(
            f"{result.baseline}: throughput={result.system_throughput_req_s:.4f} req/s, "
            f"req/$={result.per_dollar_throughput_req_per_dollar:.2f}, "
            f"p99={result.p99_latency_s:.4f}s"
        )


if __name__ == "__main__":
    main()

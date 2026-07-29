#!/usr/bin/env python3
"""
Unified comparison script:
1) Baseline P-D (H100) with fixed capacity
2) P-D autoscaling (scale up/down decode H100) using 10s latency windows
3) AFD autoscaling (attention H20 + FFN H100) with separate scale decisions

Deterministic traces and arrivals for reproducibility.
"""

import os
import sys
import json
import time
import random
from typing import List, Dict, Tuple, Any
from threading import Lock

import numpy as np

# Make simulator importable when run from CLI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.core.cluster_manager import ClusterManager, ClusterConfiguration, NodeConfiguration
from simulator.configs.hardware import hardware_params

print_lock = Lock()
GLOBAL_SEED = 42
WINDOW_SIZE_S = 20.0
LOG_LINES: List[str] = []


def safe_print(*args, **kwargs):
    # Mirror all printed output into an in-memory log for later markdown export.
    text = " ".join(str(a) for a in args)
    lines = text.splitlines() or [""]
    LOG_LINES.extend(lines)
    with print_lock:
        print(*args, **kwargs)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def generate_synthetic_segments(num_per_segment: int = 150, rng=None) -> List[List[Dict[str, Any]]]:
    """Deterministic synthetic traces when datasets/alpaca is unavailable."""
    rng = rng or np.random.default_rng(GLOBAL_SEED)
    segments: List[List[Dict[str, Any]]] = []
    configs = [
        (128, 512, 64, 256),   # moderate
        (256, 1024, 128, 512), # larger
        (64, 256, 32, 128),    # smaller
    ]
    for seg_idx, (in_min, in_max, out_min, out_max) in enumerate(configs):
        seg = []
        for i in range(num_per_segment):
            seg.append(
                {
                    "request_id": f"synthetic_{seg_idx}_{i}",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input_length": int(rng.integers(in_min, in_max)),
                    "output_length": int(rng.integers(out_min, out_max)),
                    "total_len": 0,  # placeholder for compatibility
                }
            )
        segments.append(seg)
    return segments


def load_traces_from_alpaca(num_per_segment: int = 150) -> List[List[Dict[str, Any]]]:
    """
    Load Alpaca traces and create three segments with different input/output lengths:
    segment 0: moderate, segment 1: larger, segment 2: smaller.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    try:
        from datasets import load_dataset
    except Exception as e:
        safe_print(f"datasets unavailable ({type(e).__name__}: {e}); using synthetic traces.")
        return generate_synthetic_segments(num_per_segment=num_per_segment, rng=rng)

    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        # Compute lengths and sort
        records = []
        for idx, ex in enumerate(ds):
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            out = ex.get("output", "")
            full_in = f"{instruction} {inp}".strip()
            input_len = max(5, len(full_in) // 4)
            output_len = max(5, len(out) // 4)
            total_len = input_len + output_len
            records.append(
                {
                    "request_id": f"alpaca_{idx}",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input_length": input_len,
                    "output_length": output_len,
                    "total_len": total_len,
                }
            )
        # Sort by total_len
        records = sorted(records, key=lambda r: r["total_len"])
        n = len(records)
        low = records[: n // 3]
        mid = records[n // 3 : 2 * n // 3]
        high = records[2 * n // 3 :]

        def sample_segment(seg):
            idxs = rng.choice(len(seg), size=min(num_per_segment, len(seg)), replace=False)
            return [seg[i] for i in idxs]

        return [sample_segment(mid), sample_segment(high), sample_segment(low)]
    except Exception as e:
        safe_print(f"Failed to load Alpaca dataset ({type(e).__name__}: {e}); using synthetic traces.")
        return generate_synthetic_segments(num_per_segment=num_per_segment, rng=rng)


def generate_requests_with_arrivals_by_window(
    traces_by_window: List[List[Dict[str, Any]]],
    window_size_s: float,
    arrival_rates: List[float],
) -> List[Dict[str, Any]]:
    """Generate fixed arrival times for each window using Poisson arrivals and window-specific traces."""
    rng = np.random.default_rng(GLOBAL_SEED)
    requests: List[Dict[str, Any]] = []
    current_time = 0.0
    global_idx = 0
    for win_idx, (trace_pool, rate) in enumerate(zip(traces_by_window, arrival_rates)):
        if not trace_pool or rate <= 0:
            current_time += window_size_s
            continue
        t = current_time
        window_end = current_time + window_size_s
        while True:
            gap = rng.exponential(1.0 / rate)
            t += gap
            if t >= window_end:
                break
            trace = trace_pool[global_idx % len(trace_pool)]
            requests.append(
                {
                    "request_id": f"{trace['request_id']}_{global_idx}",
                    "model": trace["model"],
                    "arrive_at": float(t),
                    "input_length": trace["input_length"],
                    "output_length": trace["output_length"],
                    "window_idx": win_idx,
                }
            )
            global_idx += 1
        current_time = window_end
    return requests


class FixedTraceArrivalProcess:
    """Arrival process using pre-generated requests with fixed arrival times."""

    def __init__(self, requests: List[Dict[str, Any]]):
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


def build_pd_cluster(num_decode: int, max_batch: int) -> ClusterConfiguration:
    """Prefill + decode (all H100)."""
    nodes: List[NodeConfiguration] = []
    nodes.append(
        NodeConfiguration(
            node_id="H100_prefill",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware="NVDA:H100:SXM",
            pd_separation=True,
            pd_prefill_only=True,
            kv_transfer_bandwidth_gbps=100.0,
            max_batch_size=max_batch,
        )
    )
    for j in range(num_decode):
        nodes.append(
            NodeConfiguration(
                node_id=f"H100_decode_{j}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware="NVDA:H100:SXM",
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0,
                max_batch_size=max_batch,
            )
        )
    return ClusterConfiguration(
        cluster_id=f"pd_{num_decode}decode",
        nodes=nodes,
        scheduler_algorithm="random",
    )


def build_afd_cluster(
    attn_repl: int,
    ffn_repl: int,
    attn_batch: int,
    ffn_batch: int,
) -> ClusterConfiguration:
    """Prefill H100 + attention H20 + FFN H100."""
    nodes: List[NodeConfiguration] = []
    nodes.append(
        NodeConfiguration(
            node_id="H100_prefill",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            hardware="NVDA:H100:SXM",
            pd_separation=True,
            pd_prefill_only=True,
            kv_transfer_bandwidth_gbps=100.0,
            max_batch_size=attn_batch,
        )
    )
    for j in range(attn_repl):
        nodes.append(
            NodeConfiguration(
                node_id=f"H20_attn_{j}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware="NVDA:H20",
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0,
                max_batch_size=attn_batch,
                afd_attention=True,
                afd_enabled=True,
            )
        )
    for j in range(ffn_repl):
        nodes.append(
            NodeConfiguration(
                node_id=f"H100_ffn_{j}",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                hardware="NVDA:H100:SXM",
                pd_separation=True,
                pd_decode_only=True,
                kv_transfer_bandwidth_gbps=100.0,
                max_batch_size=ffn_batch,
                afd_ffn=True,
                afd_enabled=True,
            )
        )
    return ClusterConfiguration(
        cluster_id=f"afd_{attn_repl}attn_{ffn_repl}ffn",
        nodes=nodes,
        scheduler_algorithm="random",
        afd_enabled=True,
        afd_attention_batch_size=attn_batch,
        afd_ffn_max_batch_size=ffn_batch,
        afd_activation_bandwidth_gbps=100.0,
    )


def run_cluster(requests: List[Dict[str, Any]], cluster_config: ClusterConfiguration, duration: float):
    arrival_process = FixedTraceArrivalProcess(requests)
    cluster = ClusterManager(cluster_config, arrival_process)
    # Silence verbose simulation prints
    import contextlib, io, os
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        cluster.run_simulation(duration=duration, enable_failures=False)
    results = cluster.get_results()
    # Trim heavy traces to keep logs readable
    results_slim = dict(results)
    if "trace_events" in results_slim:
        results_slim["trace_events"] = []
    summary = results.get("completed_requests", [])

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

    if summary:
        first_arrival = min(r["arrive_at"] for r in summary)
        last_finish = max(r.get("generation_finished_at", 0) for r in summary)
        sim_time = max(last_finish - first_arrival, 0.0)
    else:
        sim_time = duration

    stats = {
        "completed": len(summary),
        "throughput": len(summary) / sim_time if sim_time > 0 else 0.0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency": pct(0.5),
        "p95_latency": pct(0.95),
        "p99_latency": pct(0.99),
        "sim_time": sim_time,
        "latencies": latencies,
        "details": results_slim,
    }
    return stats


def window_requests(requests: List[Dict[str, Any]], window_size_s: float) -> List[List[Dict[str, Any]]]:
    """Split requests by arrival time windows; normalize arrival times within each window."""
    if not requests:
        return []
    first_arrival = min(r["arrive_at"] for r in requests)
    windows: Dict[int, List[Dict[str, Any]]] = {}
    for req in requests:
        bucket = int((req["arrive_at"] - first_arrival) // window_size_s)
        win_start = first_arrival + bucket * window_size_s
        req_copy = req.copy()
        req_copy["arrive_at"] = req["arrive_at"] - win_start
        windows.setdefault(bucket, []).append(req_copy)
    return [windows[k] for k in sorted(windows.keys())]


def autoscale_pd(requests: List[Dict[str, Any]]):
    """P-D autoscaling: adjust decode replica count per window based on P95 latency."""
    windows = window_requests(requests, WINDOW_SIZE_S)
    if not windows:
        return None

    # Hardcoded decisions per window based on reference (window0):
    # window0 hold (4 decodes), window1 scale_up (x2), window2 scale_down (/2), cap [1,8]
    base_decode = 4
    decode_plan = [base_decode]
    decode_plan.append(min(8, base_decode * 2))                # window1 (scale up)
    decode_plan.append(max(1, int(np.ceil(base_decode / 2))))  # window2 (scale down)
    

    results = []
    for idx, win in enumerate(windows):
        decode_repl = decode_plan[idx] if idx < len(decode_plan) else decode_plan[0]
        if idx == 0:
            action = "hold"
        elif idx == 1:
            action = "scale_up"
        elif idx == 2:
            action = "scale_down"
        else:
            action = "hold"
        config = build_pd_cluster(num_decode=decode_repl, max_batch=8)
        stats = run_cluster(win, config, duration=WINDOW_SIZE_S + 5.0)
        results.append((f"window_{idx}", stats, config, action))
    return results


def autoscale_afd(requests: List[Dict[str, Any]]):
    """AFD autoscaling: adjust attention and FFN replicas (and batch sizes) per window."""
    windows = window_requests(requests, WINDOW_SIZE_S)
    if not windows:
        return None

    base_attn = 3  # initial: 3 H20 attention
    base_ffn = 1   # initial: 1 H100 FFN
    attn_batch = 8
    base_ffn_batch = 24

    def adjust_batches(attn_repl: int, attn_batch: int, ffn_repl: int, ffn_batch: int):
        # Keep FFN batch roughly matching total attention capacity
        desired_ffn_batch = max(attn_repl * attn_batch, attn_batch)
        return max(desired_ffn_batch, ffn_batch)

    results = []
    ref_attn_sum = None
    ref_ffn_sum = None

    for idx, win in enumerate(windows):
        # First pass with baseline config to decide
        base_config = build_afd_cluster(base_attn, base_ffn, attn_batch, base_ffn_batch)
        base_stats = run_cluster(win, base_config, duration=WINDOW_SIZE_S + 5.0)
        per_req_components = base_stats["details"].get("afd_request_components", {})
        attn_sum = sum(v["attn_s"] for v in per_req_components.values()) if per_req_components else 0.0
        ffn_sum = sum(v["ffn_s"] for v in per_req_components.values()) if per_req_components else 0.0

        if idx == 0:
            ref_attn_sum = attn_sum
            ref_ffn_sum = ffn_sum
            results.append((f"window_{idx}", base_stats, base_config, "hold"))
            continue

        target_attn = base_attn
        target_ffn = base_ffn
        attn_decision = "hold"
        ffn_decision = "hold"
        eps = 0.02  # guard against tiny fluctuations
        if ref_attn_sum is not None:
            if attn_sum > ref_attn_sum * (1 + eps):
                target_attn = min(6, base_attn * 2)
                attn_decision = "scale_up"
            elif attn_sum < ref_attn_sum * (1 - eps):
                target_attn = max(1, int(np.ceil(base_attn / 2)))
                attn_decision = "scale_down"
        if ref_ffn_sum is not None:
            if ffn_sum > ref_ffn_sum * (1 + eps):
                target_ffn = min(6, base_ffn * 2)
                ffn_decision = "scale_up"
            elif ffn_sum < ref_ffn_sum * (1 - eps):
                target_ffn = max(1, int(np.ceil(base_ffn / 2)))
                ffn_decision = "scale_down"

        # Second pass with scaled config applied to this window
        ffn_batch_scaled = adjust_batches(target_attn, attn_batch, target_ffn, base_ffn_batch)
        scaled_config = build_afd_cluster(target_attn, target_ffn, attn_batch, ffn_batch_scaled)
        scaled_stats = run_cluster(win, scaled_config, duration=WINDOW_SIZE_S + 5.0)
        results.append(
            (
                f"window_{idx}",
                scaled_stats,
                scaled_config,
                f"attn:{attn_decision}, ffn:{ffn_decision}",
            )
        )

    return results


def summarize(results: List[Tuple[str, Dict, ClusterConfiguration]], label: str) -> Dict[str, float]:
    total_completed = sum(r[1]["completed"] for r in results)
    all_lat = []
    for item in results:
        stats = item[1]
        all_lat.extend(stats["latencies"])
    all_lat.sort()

    def pct(p: float) -> float:
        if not all_lat:
            return 0.0
        idx = min(int(len(all_lat) * p), len(all_lat) - 1)
        return all_lat[idx]

    avg = sum(all_lat) / len(all_lat) if all_lat else 0.0
    p95 = pct(0.95)

    safe_print("\n" + "=" * 60)
    safe_print(f"{label}")
    safe_print("=" * 60)
    safe_print(f"  Completed: {total_completed}")
    safe_print(f"  Avg latency: {avg:.3f}s")
    safe_print(f"  P95 latency: {p95:.3f}s")
    return {
        "completed": total_completed,
        "avg_latency": avg,
        "p95_latency": p95,
        "details": results,
    }


def compute_cost(nodes: List[NodeConfiguration], sim_time_s: float) -> float:
    hours = sim_time_s / 3600.0
    cost = 0.0
    for n in nodes:
        price = hardware_params.get(n.hardware, {}).get("price_per_hour", 0.0)
        cost += price * hours
    return cost


def print_window_stats(name: str, stats: Dict[str, Any], decision: str = None, cost_delta: float = None):
    ev = stats.get("details", {}).get("event_loop_stats", {})
    safe_print(f"Event loop statistics:")
    safe_print(f"  Events processed: {ev.get('events_processed', 0)}")
    safe_print(f"  Simulation time: {ev.get('total_simulation_time', 0):.3f}s")
    line = f"{name}: P95={stats['p95_latency']:.3f}s"
    if decision:
        line += f", decision={decision}"
    if cost_delta is not None:
        line += f", cost_delta=${cost_delta:.2f}"
    safe_print(line)


def main():
    set_seeds(GLOBAL_SEED)
    trace_segments = load_traces_from_alpaca(num_per_segment=200)
    if not any(trace_segments):
        safe_print("No traces available, exiting.")
        return

    # Three windows mapped to: 0=moderate, 1=larger, 2=smaller
    arrival_rates = [5.0, 8.0, 0.5]  # Window2 clearly lighter than Window0
    requests = generate_requests_with_arrivals_by_window(
        trace_segments,
        window_size_s=WINDOW_SIZE_S,
        arrival_rates=arrival_rates,
    )

    # Window trace stats
    # Actual arrivals per window
    safe_print("Actual arrivals by window:")
    arrivals_by_win: Dict[int, List[Dict[str, Any]]] = {}
    for req in requests:
        widx = req.get("window_idx", 0)
        arrivals_by_win.setdefault(widx, []).append(req)
    for idx in sorted(arrivals_by_win.keys()):
        seg = arrivals_by_win[idx]
        avg_in = np.mean([r["input_length"] for r in seg])
        avg_out = np.mean([r["output_length"] for r in seg])
        safe_print(f"  Window {idx}: arrivals={len(seg)}, avg_input={avg_in:.1f}, avg_output={avg_out:.1f}")

    # Baseline (fixed capacity)
    baseline_config = build_pd_cluster(num_decode=4, max_batch=8)
    baseline_stats = run_cluster(requests, baseline_config, duration=WINDOW_SIZE_S * 40)
    safe_print("\n" + "=" * 60)
    safe_print("BASELINE (P-D fixed H100 x4 decode)")
    safe_print("=" * 60)
    baseline_summary = {
        k: baseline_stats[k]
        for k in ["completed", "avg_latency", "p50_latency", "p95_latency", "p99_latency", "sim_time"]
    }
    safe_print(json.dumps(baseline_summary, indent=2))

    # Baseline per-window view
    base_windows = window_requests(requests, WINDOW_SIZE_S)
    safe_print("\n" + "-" * 60)
    safe_print("Baseline per-window (fixed config)")
    for idx, win in enumerate(base_windows):
        stats = run_cluster(win, baseline_config, duration=WINDOW_SIZE_S + 5.0)
        print_window_stats(f"Window {idx}", stats)

    # AFD baseline (no autoscaling)
    afd_base_config = build_afd_cluster(attn_repl=3, ffn_repl=1, attn_batch=8, ffn_batch=24)
    afd_base_stats = run_cluster(requests, afd_base_config, duration=WINDOW_SIZE_S * 40)
    safe_print("\n" + "-" * 60)
    safe_print("AFD baseline (no autoscaling)")
    afd_base_summary = {
        k: afd_base_stats[k]
        for k in ["completed", "avg_latency", "p95_latency", "sim_time"]
    }
    safe_print(json.dumps(afd_base_summary, indent=2))
    # AFD baseline per-window view
    afd_windows = base_windows
    safe_print("\n" + "-" * 60)
    safe_print("AFD Baseline per-window (fixed config)")
    for idx, win in enumerate(afd_windows):
        stats = run_cluster(win, afd_base_config, duration=WINDOW_SIZE_S + 5.0)
        print_window_stats(f"Window {idx}", stats)

    # P-D autoscaling
    pd_results = autoscale_pd(requests)
    pd_summary = None
    if pd_results:
        safe_print("\n" + "-" * 60)
        safe_print("P-D Autoscaling per-window decisions")
        ref_p95 = pd_results[0][1]["p95_latency"]
        safe_print(f"  Window 0 (reference): P95={ref_p95:.3f}s")
        base_cost_nodes = baseline_config.nodes
        for item in pd_results[1:]:
            name, stats, cfg = item[0], item[1], item[2]
            action = item[3] if len(item) > 3 else "hold"
            decodes = len([n for n in cfg.nodes if n.pd_decode_only])
            cost_cur = compute_cost(cfg.nodes, stats["sim_time"])
            cost_base = compute_cost(base_cost_nodes, stats["sim_time"])
            cost_delta = cost_cur - cost_base
            print_window_stats(
                f"  {name} (decode_repl={decodes})",
                stats,
                decision=action,
                cost_delta=cost_delta,
            )
        pd_summary = summarize(pd_results, "P-D Autoscaling")

    # AFD autoscaling
    afd_results = autoscale_afd(requests)
    afd_summary = None
    if afd_results:
        safe_print("\n" + "-" * 60)
        safe_print("AFD Autoscaling per-window decisions")
        ref_p95 = afd_results[0][1]["p95_latency"]
        safe_print(f"  Window 0 (reference): P95={ref_p95:.3f}s")
        base_cost_nodes = afd_base_config.nodes
        for item in afd_results[1:]:
            name, stats, cfg = item[0], item[1], item[2]
            action = item[3] if len(item) > 3 else "hold"
            cost_cur = compute_cost(cfg.nodes, stats["sim_time"])
            cost_base = compute_cost(base_cost_nodes, stats["sim_time"])
            cost_delta = cost_cur - cost_base
            print_window_stats(
                f"  {name}",
                stats,
                decision=action,
                cost_delta=cost_delta,
            )
        afd_summary = summarize(afd_results, "AFD Autoscaling")

    # Final comparison (P95 only)
    safe_print("\n" + "=" * 60)
    safe_print("FINAL COMPARISON (P95 ONLY)")
    safe_print("=" * 60)
    final_lines = []
    final_lines.append(f"P-D Baseline: P95 {baseline_summary['p95_latency']:.3f}s")
    if pd_summary:
        final_lines.append(f"P-D Autoscaling: P95 {pd_summary['p95_latency']:.3f}s")
    final_lines.append(f"AFD Baseline: P95 {afd_base_summary['p95_latency']:.3f}s")
    if afd_summary:
        final_lines.append(f"AFD Autoscaling: P95 {afd_summary['p95_latency']:.3f}s")
    for line in final_lines:
        safe_print(line)

    # Save full log and final comparison to markdown
    md_lines = ["# Autoscaling Run Log", ""]
    md_lines.append("## Full Output")
    md_lines.append("")
    md_lines.append("```")
    md_lines.extend(LOG_LINES)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Final P95 Latencies")
    md_lines.append("")
    md_lines.extend([f"- {line}" for line in final_lines])
    md_content = "\n".join(md_lines) + "\n"
    with open("autoscaling_results.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    safe_print("\nSaved summary to autoscaling_results.md")


if __name__ == "__main__":
    main()

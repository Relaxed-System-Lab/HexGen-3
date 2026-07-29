# HexGen-3 Scheduling Framework

This repo is a copy of `llm-serving-simulator-main` with an additive
`simulator.scheduling` package that implements the scheduling framework from
`694_HexGen_3_A_Fully_Disaggreg (5).pdf`.

## Implemented Components

- Workload profile `W = <arrival_rate, input_length_distribution, output_length_distribution>`.
- Allocation matrix `A[worker_type][hardware]` for `pre`, `attn`, and `ffn` workers.
- Local scheduler that enumerates valid replica-aware `(DP, TP, EP)` strategies for
  each non-zero allocation slice. It supports non-uniform TP partitions where
  `sum(replica.tp) == allocated_gpus`, plus the paper's intra-replica and
  cross-replica EP constraints.
- Global scheduler using guided simulated annealing. It moves GPU blocks from
  high-throughput worker types or an `empty` source of newly available GPUs to
  bottleneck worker types, and accepts candidate allocations by
  bottleneck-throughput reward.
- Model-size-derived GPU block sizes when `--block-size 0` is used.
- Warm-start projection that preserves TP/EP and scales DP when possible.
- Cost-aware local search using a stability-window reload-cost amortization test.
- Autoscaling-aware rescheduling helpers with proportional per-worker scaling.
- Simulator-backed estimator using the existing roofline `ModelAnalyzer`, with decode
  decomposed into attention and FFN components and per-replica timing.
- Cost, bottleneck throughput, per-worker throughput, mean latency, and p50/p95/p99
  queueing latency summaries.
- Conversion helpers for simulator AFD and PD `ClusterConfiguration` objects.
- AFD simulator integration now passes per-instance attention and FFN analyzers so
  heterogeneous attention/FFN hardware is not collapsed to the first attention node.

## Run

```bash
python3 cli/run_hexgen3_scheduler.py \
  --capacity '{"NVDA:H100:SXM": 4, "NVDA:H20": 4}' \
  --arrival-rate 4 \
  --long-ratio 0.3 \
  --model-size-billions 70 \
  --cost-aware \
  --iterations 40 \
  --output output_test/hexgen3_plan.json
```

The output JSON contains the selected allocation, parallelism strategy, worker
throughputs, system bottleneck throughput, estimated latency, tail latencies, hourly
cost, and the simulated annealing history.

## Notes

The scheduler is designed for comparative evaluation. The inner loop still uses the
copied simulator's analytical model, but it now samples the workload distribution to
produce queueing tail-latency estimates and uses per-instance AFD analyzers for
heterogeneous hardware. This is still an estimator, not a cycle-accurate distributed
runtime.
